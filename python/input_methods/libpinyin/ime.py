from textService import TextService, KeyEvent, TF_MOD_SHIFT, TF_MOD_CONTROL
from keycodes import *
from . import opencc
from .libpinyin_consts import *
from .libpinyin import PY, GLIB
from ctypes import (
    c_uint, c_char, POINTER, byref, c_void_p, string_at, c_int, c_uint8,
)
from os.path import expandvars
import sys
import json
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional
from bisect import bisect_left
from itertools import chain
import functools
from threading import RLock, Thread, Event


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def modifier_down_count(keyEvent: KeyEvent):
    return sum(keyEvent.isKeyDown(k) for k in [
        VK_LSHIFT, VK_LCONTROL, VK_LMENU, VK_LWIN,
        VK_RSHIFT, VK_RCONTROL, VK_RMENU, VK_RWIN])


def char_from_key_event(keyEvent: KeyEvent):
    return (chr(keyEvent.charCode)
            if keyEvent.charCode and not keyEvent.isKeyDown(VK_CONTROL)
            else '')


def merge_config(dest: dict, src: dict):
    for k, v in src.items():
        if (isinstance(dest.get(k), dict) and isinstance(v, dict)):
            merge_config(dest[k], v)
        else:
            dest[k] = v


@dataclass
class Candidate:
    value: str
    lookup_candidate: c_void_p
    is_nbest: bool


FULL_WIDTH_ASCII_DICT = {i: i + 0xFEE0 for i in range(0x21, 0x7F)}
FULL_WIDTH_ASCII_DICT[0x20] = 0x3000

GUID_SHIFT_SPACE = '{6f0b6fac-fa94-4eb4-aea8-86e95e91ec1a}'
GUID_CTRL_PERIOD = '{54e6c3b0-afdb-430a-92b1-76259247570f}'


def synchronized(lock: RLock):
    def _decorator(wrapped):
        @functools.wraps(wrapped)
        def _wrapper(*args, **kwargs):
            with lock:
                return wrapped(*args, **kwargs)
        return _wrapper
    return _decorator


_base_dir = Path(__file__).parent.resolve()
_config_dir = Path(expandvars('%APPDATA%')) / 'PIME/libpinyin'
_userdata_dir = _config_dir / 'userdata'
_userdata_dir.mkdir(exist_ok=True, parents=True)
_context = PY.pinyin_init(str(_base_dir / 'data').encode(),
                          str(_userdata_dir).encode())
_context_lock = RLock()
_save_context_event = Event()


def _save_context_thread():
    while True:
        _save_context_event.wait()
        _save_context_event.clear()
        while _save_context_event.wait(timeout=5*60):
            _save_context_event.clear()
        with _context_lock:
            PY.pinyin_save(_context)


Thread(target=_save_context_thread).start()


class IMETextService(TextService):
    PAIR_KEYS = {
        ',': ('comma_period', False),
        '.': ('comma_period', True),
        '-': ('minus_equal', False),
        '=': ('minus_equal', True),
        '[': ('square_brackets', False),
        ']': ('square_brackets', True),
    }

    @synchronized(_context_lock)
    def onActivate(self):
        super().onActivate()
        self._modifier_press_state = ''
        self._number_decimal_state = False
        self._punctuation_indexes = defaultdict(int)

        self._config = {}
        for filename in [_base_dir / 'defaults.json',
                         _config_dir / 'config.json']:
            if Path(filename).is_file():
                with open(filename, 'r', encoding='utf-8') as f:
                    merge_config(self._config, json.load(f))
        self._enabled = self._config['enabled']
        self._punctuation = self._config['punctuation']
        self._full_width_ascii = self._config['full_width_ascii']
        self._punctuation_enabled = self._config['punctuation_enabled']
        self._custom_phrases_keys = list(self._config['custom_phrases'].keys())
        self._custom_phrases_keys.sort()

        options = (PINYIN_INCOMPLETE | DYNAMIC_ADJUST |
                   USE_DIVIDED_TABLE | USE_RESPLIT_TABLE)
        for ambiguity in self._config['pinyin_ambiguities']:
            options |= {
                'all': PINYIN_AMB_ALL,
                'c_ch': PINYIN_AMB_C_CH, 's_sh': PINYIN_AMB_S_SH,
                'z_zh': PINYIN_AMB_Z_ZH, 'f_h': PINYIN_AMB_F_H,
                'g_k': PINYIN_AMB_G_K, 'l_n': PINYIN_AMB_L_N,
                'l_r': PINYIN_AMB_L_R, 'an_ang': PINYIN_AMB_AN_ANG,
                'en_eng': PINYIN_AMB_EN_ENG, 'in_ing': PINYIN_AMB_IN_ING,
            }[ambiguity]
        for correction in self._config['pinyin_corrections']:
            options |= {
                'all': PINYIN_CORRECT_ALL,
                'gn_ng': PINYIN_CORRECT_GN_NG, 'mg_ng': PINYIN_CORRECT_MG_NG,
                'iou_iu': PINYIN_CORRECT_IOU_IU, 'uei_ui': PINYIN_CORRECT_UEI_UI,
                'uen_un': PINYIN_CORRECT_UEN_UN, 'ue_ve': PINYIN_CORRECT_UE_VE,
                'v_u': PINYIN_CORRECT_V_U, 'on_ong': PINYIN_CORRECT_ON_ONG,
            }[correction]
        PY.pinyin_set_options(_context, options)
        if self._config["double_pinyin_scheme"]:
            PY.pinyin_set_double_pinyin_scheme(_context, {
                'zrm': DOUBLE_PINYIN_ZRM, 'ms': DOUBLE_PINYIN_MS,
                'ziguang': DOUBLE_PINYIN_ZIGUANG, 'abc': DOUBLE_PINYIN_ABC,
                'pyjj': DOUBLE_PINYIN_PYJJ, 'xhe': DOUBLE_PINYIN_XHE,
            }[self._config["double_pinyin_scheme"]])
            self._pinyin_get_pinyin_auxiliary_text = (
                PY.pinyin_get_double_pinyin_auxiliary_text)
            self._pinyin_parse_more_pinyins = (
                PY.pinyin_parse_more_double_pinyins)
        else:
            self._pinyin_get_pinyin_auxiliary_text = (
                PY.pinyin_get_full_pinyin_auxiliary_text)
            self._pinyin_parse_more_pinyins = (
                PY.pinyin_parse_more_full_pinyins)

        self._load_dicts(_config_dir / 'dict',
                         _userdata_dir / '.dict.sum')
        self._instance = PY.pinyin_alloc_instance(_context)

        self._opencc = (
            opencc.OpenCC({
                's': 't2s', 't': 's2t',
            }[self._config['chinese_character_conversion']])
        ) if self._config['chinese_character_conversion'] else None

        self.customizeUI(
            candFontName=self._config['classic_ui_font_name'],
            candFontSize=self._config['classic_ui_font_size'],
            candPerRow=self._config['classic_ui_candidates_per_row'],
            candUseCursor=self._config['classic_ui_show_candidate_cursor'],
        )
        self.setSelKeys("123456789")
        if self.client.isWindows8Above:
            if self._config['show_mode_icon']:
                self.addButton("windows-mode-icon", commandId=1)
            else:
                self.removeButton("windows-mode-icon")
            self._update_mode_icon()

        if self._config['shift_space_toggle_full_width_ascii']:
            self.addPreservedKey(VK_SPACE, TF_MOD_SHIFT, GUID_SHIFT_SPACE)
        else:
            self.removePreservedKey(GUID_SHIFT_SPACE)
        if self._config['ctrl_period_toggle_punctuation_enabled']:
            self.addPreservedKey(
                VK_OEM_PERIOD, TF_MOD_CONTROL, GUID_CTRL_PERIOD)
        else:
            self.removePreservedKey(GUID_CTRL_PERIOD)

        self._reset()

    @synchronized(_context_lock)
    def onDeactivate(self):
        self.removePreservedKey(GUID_SHIFT_SPACE)
        if self.client.isWindows8Above:
            self.removeButton("windows-mode-icon")

        del self._opencc
        PY.pinyin_free_instance(self._instance)
        super().onDeactivate()

    def _update_mode_icon(self):
        name = 'enabled' if self._enabled else 'disabled'
        self.changeButton("windows-mode-icon",
                          tooltip="中文" if self._enabled else '英文',
                          icon=str(_base_dir / f"icon/{name}.ico"))

    def _toggle_enabled(self):
        if self._enabled:
            self._commit()
        self._enabled ^= True
        self._update_mode_icon()

    def filterKeyDown(self, keyEvent: KeyEvent):
        char = char_from_key_event(keyEvent)
        self._number_decimal_state = not self._input and (
            '0' <= char <= '9' or self._number_decimal_state and char == '.')

        if keyEvent.keyCode == VK_SHIFT or keyEvent.keyCode == VK_CONTROL:
            if (self._modifier_press_state
                    or modifier_down_count(keyEvent) <= 1):
                self._modifier_press_state += (
                    's' if keyEvent.keyCode == VK_SHIFT else 'c')
            else:
                self._modifier_press_state = ''
            return False
        self._modifier_press_state = ''

        if self._full_width_ascii and keyEvent.isPrintableChar():
            return True

        if not self._enabled:
            return False

        if not self._input:
            return ('a' <= char <= 'z' or 'A' <= char <= 'Z' or
                    char in self._punctuation and self._punctuation_enabled)
        return True

    def filterKeyUp(self, keyEvent: KeyEvent):
        if keyEvent.keyCode == VK_SHIFT or keyEvent.keyCode == VK_CONTROL:
            if not self._modifier_press_state:
                return False
            state = self._modifier_press_state + (
                'S' if keyEvent.keyCode == VK_SHIFT else 'C')
            if modifier_down_count(keyEvent) == 0:
                if (
                    self._config['shift_toggle_enabled'] and state == 'sS'
                ) or (
                    self._config['ctrl_toggle_enabled'] and state == 'cC'
                ) or (
                    self._config['ctrl_shift_toggle_enabled'] and
                    len(state) == 4 and state[:2] in ['sc', 'cs'] and
                    state[2:] in ['SC', 'CS']
                ):
                    # do not change _modifier_press_state; may be called again
                    return True
                self._modifier_press_state = ''
                return False
            else:
                self._modifier_press_state = state
                return False
        self._modifier_press_state = ''
        return False

    def _show_composition(self, parsed_input_len: int):
        if self._config['original_composition_string']:
            composition = self._input[self._partial_pos:]
            cursor = self._input_pos - self._partial_pos
        else:
            p = POINTER(c_char)()
            self._pinyin_get_pinyin_auxiliary_text(
                self._instance, self._partial_pos, byref(p))
            prefix = string_at(p).decode().split('|')[0]
            GLIB.g_free(p)
            self._pinyin_get_pinyin_auxiliary_text(
                self._instance, self._input_pos, byref(p))
            composition_split = string_at(p).decode().split('|')
            GLIB.g_free(p)
            composition = ''.join(composition_split).rstrip()
            cursor = min(len(composition_split[0]), len(composition))
            if composition.startswith(prefix):
                composition = composition[len(prefix):]
                cursor -= len(prefix)
            if parsed_input_len < len(self._input):
                if composition:
                    composition += ' '
                s = self._input[parsed_input_len:]
                i = self._input_pos - parsed_input_len
                if i >= 0:
                    cursor = len(composition) + i
                composition += s
        self.setCompositionString(self._partial + composition)
        self.setCompositionCursor(len(self._partial) + cursor)

    @synchronized(_context_lock)
    def _get_candidate_at(self, index: int):
        lc = c_void_p()
        PY.pinyin_get_candidate(self._instance, index, byref(lc))
        s = POINTER(c_char)()
        PY.pinyin_get_candidate_string(self._instance, lc, byref(s))
        value = string_at(s).decode()
        t = c_int()
        PY.pinyin_get_candidate_type(self._instance, lc, byref(t))
        is_nbest = t.value == NBEST_MATCH_CANDIDATE
        if is_nbest and value.startswith(self._partial):
            value = value[len(self._partial):]

        if self._opencc:
            value = self._opencc.convert(value)
        return Candidate(value=value, is_nbest=is_nbest,
                         lookup_candidate=lc)

    def _show_candidates(self):
        candidate_list = [c.value for c in self._candidates[
            self._candidate_start:
            self._candidate_start + self._config['max_candidates']]]
        self.setCandidateList(candidate_list)
        self.setShowCandidates(bool(candidate_list))
        self.setCandidateCursor(0)

    @synchronized(_context_lock)
    def _update_input(self, back_deselect: bool):
        if not self._input:
            return self._reset()

        self._back_deselect = back_deselect
        parsed_input_len = self._pinyin_parse_more_pinyins(
            self._instance, self._input.encode())
        PY.pinyin_guess_sentence(self._instance)
        if 0 < self._partial_pos >= parsed_input_len:
            return self._commit()

        custom_phrases = []
        if (not self._partial_pos and
                self._input in self._config["custom_phrases"]):
            custom_phrases = self._config["custom_phrases"][self._input]
            if isinstance(custom_phrases, str):
                return self._commit(append=custom_phrases, rest=False)

        PY.pinyin_guess_candidates(
            self._instance, self._partial_pos,
            SORT_BY_PHRASE_LENGTH_AND_PINYIN_LENGTH_AND_FREQUENCY)
        n = c_uint()
        PY.pinyin_get_n_candidate(self._instance, byref(n))
        candidates = [self._get_candidate_at(i) for i in range(n.value)]
        value_set = set()
        self._candidates = [v for v in candidates if not (
            v.value in value_set or value_set.add(v.value))]
        self._merge_custom_phrases(custom_phrases)
        self._candidate_start = 0
        self._show_composition(parsed_input_len)
        self._show_candidates()

    def _reset(self):
        self._input = ''
        self._input_pos = 0
        self._partial = ''
        self._partial_pos = 0
        self._candidates = []
        self._candidate_start = 0
        self._partial_history = []
        self._back_deselect = False
        self.setCompositionString('')
        self._show_candidates()

    def onCompositionTerminated(self, *args):
        super().onCompositionTerminated(*args)
        self._reset()

    def onKeyDown(self, keyEvent: KeyEvent):
        vk = keyEvent.keyCode
        char = char_from_key_event(keyEvent)

        if VK_NUMPAD0 <= vk <= VK_NUMPAD9:
            if not keyEvent.isKeyToggled(VK_NUMLOCK):
                vk = [VK_INSERT, VK_END, VK_DOWN, VK_NEXT,
                      VK_LEFT, 0, VK_RIGHT,
                      VK_HOME, VK_UP, VK_PRIOR][vk - VK_NUMPAD0]
        elif vk == VK_DECIMAL and not keyEvent.isKeyToggled(VK_NUMLOCK):
            vk = VK_DELETE

        if self._enabled and (
            'a' <= char <= 'z' or 'A' <= char <= 'Z' or
            self._input and (char == ';' or char == '\'') or
            (char and not self._partial_pos and
             self._is_custom_phrase_prefix(self._input + char))
        ):
            self._input = (self._input[:self._input_pos] + char +
                           self._input[self._input_pos:])
            self._input_pos += 1
            self._update_input(False)
            return True

        if self._input and (char == ' ' or '1' <= char <= '9'):
            n = None if char == ' ' else ord(char) - ord('1')
            self._choose(n)
            return True

        if self._input and char in self.PAIR_KEYS:
            config_key, is_next = self.PAIR_KEYS[char]
            if self._config[config_key] == 'page':
                self._turn_page(is_next)
                return True
            if self._config[config_key] == 'affix':
                candidate = self._chosen_candidate(None)
                if candidate and candidate.value:
                    s = candidate.value[-1 if is_next else 0]
                    self._commit(append=s, rest=False)
                    return True

        if self._enabled:
            if not self._input and char == '.' and self._number_decimal_state:
                self._number_decimal_state = False
                self._commit(append='.')
                return True

            if char in self._punctuation and self._punctuation_enabled:
                value = self._punctuation[char]
                if isinstance(value, list):
                    index = self._punctuation_indexes[char]
                    self._punctuation_indexes[char] = (index + 1) % len(value)
                    value = value[index]
                self._choose(None, append=value)
                return True

        if self._full_width_ascii and keyEvent.isPrintableChar():
            self._choose(None, append=char)
            return True

        if char == '\x1B':
            self._reset()
            return True

        if char == '\r':
            self._commit()
            return True

        if char == '\b':
            if self._back_deselect and self._partial_pos > 0:
                self._deselect()
                self._update_input(True)
                return True
            if self._input_pos <= 0:
                return True
            self._deselect_if_required()
            self._input = (self._input[:self._input_pos-1] +
                           self._input[self._input_pos:])
            self._input_pos -= 1
            self._update_input(False)
            return True

        if vk == VK_DELETE:
            if self._input_pos < len(self._input):
                self._input = (self._input[:self._input_pos] +
                               self._input[self._input_pos+1:])
                self._update_input(False)
            return True

        if vk == VK_LEFT or vk == VK_RIGHT:
            if vk == VK_LEFT:
                self._deselect_if_required()
            self._input_pos += 1 if vk == VK_RIGHT else -1
            self._input_pos = sorted((0, self._input_pos, len(self._input)))[1]
            self._update_input(False)
            return True

        if vk == VK_PRIOR or vk == VK_NEXT:
            self._turn_page(vk == VK_NEXT)
            return True

        if vk == VK_DOWN:
            if self.candidateCursor >= len(self.candidateList) - 1:
                self._turn_page(True)
            else:
                self.setCandidateCursor(self.candidateCursor + 1)

        if vk == VK_UP:
            if self.candidateCursor <= 0:
                self._turn_page(False, choose_last=True)
            else:
                self.setCandidateCursor(self.candidateCursor - 1)

        return True

    def onKeyUp(self, keyEvent: KeyEvent):
        vk = keyEvent.keyCode
        if vk == VK_SHIFT or vk == VK_CONTROL:
            self._modifier_press_state = ''
            self._toggle_enabled()
            return True
        return True

    @synchronized(_context_lock)
    def _deselect(self):
        self._partial, self._partial_pos = self._partial_history.pop()
        PY.pinyin_clear_constraint(self._instance, self._partial_pos)

    def _deselect_if_required(self):
        if self._input_pos == self._partial_pos > 0:
            self._deselect()

    def _commit(self, append='', rest=True):
        s = self._partial + (
            self._input[self._partial_pos:] if rest else '') + append
        if self._full_width_ascii:
            s = s.translate(FULL_WIDTH_ASCII_DICT)
        self.setCommitString(s)
        self._reset()

    def _chosen_candidate(self, n: Optional[int]):
        try:
            return self._candidates[self._candidate_start +
                                    (self.candidateCursor if n is None else n)]
        except IndexError:
            return None

    @synchronized(_context_lock)
    def _choose(self, n: Optional[int], append=''):
        candidate = self._chosen_candidate(n)
        if not candidate:
            if n is None:
                self._commit(append=append)
            return
        if not candidate.lookup_candidate:  # custom phrase
            self._commit(append=candidate.value + append, rest=False)
            return

        lookup_cursor = PY.pinyin_choose_candidate(
            self._instance, 0, candidate.lookup_candidate)
        self._partial_history.append((self._partial, self._partial_pos))
        self._partial += candidate.value
        self._partial_pos += lookup_cursor
        self._input_pos = max(self._input_pos, self._partial_pos)

        if candidate.is_nbest:
            index = c_uint8()
            PY.pinyin_get_candidate_nbest_index(
                self._instance, candidate.lookup_candidate, byref(index))
            PY.pinyin_train(self._instance, index.value)
            _save_context_event.set()
        elif self._partial_pos == len(self._input):
            PY.pinyin_guess_sentence(self._instance)
            PY.pinyin_train(self._instance, 0)
            _save_context_event.set()
        else:
            self._update_input(True)
            return
        self._commit(append=append)

    def _turn_page(self, is_next: bool, choose_last=False):
        n = self._config['max_candidates']
        p = max(0, self._candidate_start + (n if is_next else -n))
        if p < len(self._candidates) and p != self._candidate_start:
            self._candidate_start = p
            self._show_candidates()
            self.setCandidateCursor(0 if not choose_last else
                                    max(len(self.candidateList) - 1, 0))

    def _merge_custom_phrases(self, custom_phrases):
        i = 0
        for s in custom_phrases:
            if isinstance(s, int):
                i = max(s - 1, 0)
            else:
                self._candidates.insert(
                    i,
                    Candidate(value=s, is_nbest=False, lookup_candidate=None))
                i += 1

    def _is_custom_phrase_prefix(self, s):
        i = bisect_left(self._custom_phrases_keys, s)
        return (i != len(self._custom_phrases_keys) and
                self._custom_phrases_keys[i].startswith(s))

    @synchronized(_context_lock)
    def _load_dicts(self, dict_dir: Path, summary_file: Path):
        dict_dir.mkdir(exist_ok=True, parents=True)
        paths = sorted(Path(dict_dir).glob('*.txt'), key=lambda x: x.name)
        summary = ''.join(f'{v.stat().st_mtime_ns} {v.name}\n' for v in paths)
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf8') as f:
                if f.read() == summary:
                    return
        token = PHRASE_INDEX_MAKE_TOKEN(NETWORK_DICTIONARY, null_token)
        PY.pinyin_mask_out(_context, PHRASE_INDEX_LIBRARY_MASK, token)

        p_imp = PY.pinyin_begin_add_phrases(_context, NETWORK_DICTIONARY)
        for path in paths:
            with open(path, 'r', encoding='utf8') as f:
                for line_lf in f:
                    phrase, pinyin, count_s, *_ = chain(
                        line_lf.strip().split(' '), [''] * 3)
                    if pinyin:
                        PY.pinyin_iterator_add_phrase(
                            p_imp, phrase.encode(), pinyin.encode(),
                            int(count_s) if count_s.isdigit() else -1)
        PY.pinyin_end_add_phrases(p_imp)

        PY.pinyin_save(_context)
        with open(summary_file, 'w', encoding='utf8') as f:
            f.write(summary)

    def onCommand(self, commandId, commandType):
        print("onCommand", commandId, commandType)

    def onPreservedKey(self, guid):
        if guid == GUID_SHIFT_SPACE:
            self._full_width_ascii ^= True
            return True
        if guid == GUID_CTRL_PERIOD:
            self._punctuation_enabled ^= True
            return True
        return False
