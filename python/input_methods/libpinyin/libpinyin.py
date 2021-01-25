from ctypes import *
from pathlib import Path

_lib_dir = Path(__file__).parent.resolve() / 'lib/libpinyin'

GLIB = CDLL(str(_lib_dir / "libglib-2.0-0.dll"))

GLIB.g_free.argtypes = [c_void_p]
GLIB.g_free.restype = c_void_p

PY = CDLL(str(_lib_dir / "pinyin.dll"))

PY.pinyin_init.argtypes = [c_char_p, c_char_p]
PY.pinyin_init.restype = c_void_p

PY.pinyin_set_options.argtypes = [c_void_p, c_uint]
PY.pinyin_set_options.restype = c_bool

PY.pinyin_alloc_instance.argtypes = [c_void_p]
PY.pinyin_alloc_instance.restype = c_void_p

PY.pinyin_parse_more_full_pinyins.argtypes = [c_void_p, c_char_p]
PY.pinyin_parse_more_full_pinyins.restype = c_size_t

PY.pinyin_parse_more_double_pinyins.argtypes = [c_void_p, c_char_p]
PY.pinyin_parse_more_double_pinyins.restype = c_size_t

PY.pinyin_guess_sentence.argtypes = [c_void_p]
PY.pinyin_guess_sentence.restype = c_bool

PY.pinyin_guess_sentence_with_prefix.argtypes = [c_void_p, c_char_p]
PY.pinyin_guess_sentence_with_prefix.restype = c_bool

PY.pinyin_guess_candidates.argtypes = [c_void_p, c_size_t, c_int]
PY.pinyin_guess_candidates.restype = c_bool

PY.pinyin_get_n_candidate.argtypes = [c_void_p, POINTER(c_uint)]
PY.pinyin_get_n_candidate.restype = c_bool

PY.pinyin_get_candidate.argtypes = [c_void_p, c_uint, POINTER(c_void_p)]
PY.pinyin_get_candidate.restype = c_bool

PY.pinyin_get_candidate_type.argtypes = [c_void_p, c_void_p, POINTER(c_int)]
PY.pinyin_get_candidate_type.restype = c_bool

PY.pinyin_get_candidate_string.argtypes = [
    c_void_p, c_void_p, POINTER(POINTER(c_char))]
PY.pinyin_get_candidate_string.restype = c_bool

PY.pinyin_get_candidate_nbest_index.argtypes = [
    c_void_p, c_void_p, POINTER(c_uint8)]
PY.pinyin_get_candidate_nbest_index.restype = c_bool

PY.pinyin_load_addon_phrase_library.argtypes = [c_void_p, c_uint8]
PY.pinyin_load_addon_phrase_library.restype = c_bool

PY.pinyin_unload_addon_phrase_library.argtypes = [c_void_p, c_uint8]
PY.pinyin_unload_addon_phrase_library.restype = c_bool

PY.pinyin_get_full_pinyin_auxiliary_text.argtypes = [
    c_void_p, c_size_t, POINTER(POINTER(c_char))]
PY.pinyin_get_full_pinyin_auxiliary_text.restype = c_bool

PY.pinyin_get_double_pinyin_auxiliary_text.argtypes = [
    c_void_p, c_size_t, POINTER(POINTER(c_char))]
PY.pinyin_get_double_pinyin_auxiliary_text.restype = c_bool

PY.pinyin_set_double_pinyin_scheme.argtypes = [c_void_p, c_int]
PY.pinyin_set_double_pinyin_scheme.restype = c_bool

PY.pinyin_choose_candidate.argtypes = [c_void_p, c_size_t, c_void_p]
PY.pinyin_choose_candidate.restype = c_int

PY.pinyin_get_pinyin_key_rest.argtypes = [
    c_void_p, c_size_t, POINTER(c_void_p)]
PY.pinyin_get_pinyin_key_rest.restype = c_bool

PY.pinyin_get_pinyin_key_rest_positions.argtypes = [
    c_void_p, c_void_p, POINTER(c_uint16), POINTER(c_uint16)]
PY.pinyin_get_pinyin_key_rest_positions.restype = c_bool

PY.pinyin_train.argtypes = [c_void_p, c_uint8]
PY.pinyin_train.restype = c_bool

PY.pinyin_clear_constraint.argtypes = [c_void_p, c_size_t]
PY.pinyin_clear_constraint.restype = c_bool

PY.pinyin_free_instance.argtypes = [c_void_p]
PY.pinyin_free_instance.restype = None

PY.pinyin_fini.argtypes = [c_void_p]
PY.pinyin_fini.restype = None

PY.pinyin_begin_add_phrases.argtypes = [c_void_p, c_uint8]
PY.pinyin_begin_add_phrases.restype = c_void_p

PY.pinyin_iterator_add_phrase.argtypes = [c_void_p, c_char_p, c_char_p, c_int]
PY.pinyin_iterator_add_phrase.restype = c_bool

PY.pinyin_end_add_phrases.argtypes = [c_void_p]
PY.pinyin_end_add_phrases.restype = None

PY.pinyin_save.argtypes = [c_void_p]
PY.pinyin_save.restype = c_bool

PY.pinyin_mask_out.argtypes = [c_void_p, c_uint32, c_uint32]
PY.pinyin_mask_out.restype = c_bool
