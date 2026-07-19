#ifndef RLLAMA_COMMON_H
#define RLLAMA_COMMON_H

#include <stdbool.h>
#include <stdint.h>

#if defined(_WIN32) && defined(RLC_BUILD)
#define RLC_API __declspec(dllexport)
#elif defined(_WIN32)
#define RLC_API __declspec(dllimport)
#else
#define RLC_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles
typedef struct rlc_chat_templates rlc_chat_templates;
typedef struct rlc_sampler rlc_sampler;

// llama.cpp handles created by the host application through libllama.
// Both libraries must link the same libllama build.
typedef struct llama_model llama_model;
typedef struct llama_context llama_context;

// Free any string returned by this library.
RLC_API void rlc_free(char * ptr);

// Create chat templates. model provides vocab BOS/EOS handling (may be NULL);
// template_src overrides the model's template when non-empty. On failure
// returns NULL and sets *err to a message (free with rlc_free).
RLC_API rlc_chat_templates * rlc_chat_templates_init(const llama_model * model,
                                             const char * template_src,
                                             const char * bos_token,
                                             const char * eos_token,
                                             char ** err);

RLC_API void rlc_chat_templates_free(rlc_chat_templates * tmpls);

// inputs_json: {"messages": [...oai...], "tools": [...oai...]|null,
//               "add_generation_prompt": bool, "enable_thinking": bool,
//               "reasoning_format": int (0=none, 1=auto: parser extracts
//               reasoning_content; must be set at apply time),
//               "parallel_tool_calls": bool}
// Returns JSON (free with rlc_free):
//   {"prompt": str, "format": int, "parser": str, "grammar": str,
//    "grammar_lazy": bool,
//    "grammar_triggers": [{"type": int, "value": str, "token": int}],
//    "generation_prompt": str,
//    "additional_stops": [str], "thinking_start_tag": str, "thinking_end_tag": str}
// On failure returns NULL and sets *err.
RLC_API char * rlc_chat_apply(rlc_chat_templates * tmpls, const char * inputs_json, char ** err);

// params_json: {"format": int, "parser": str, "generation_prompt": str,
//               "reasoning_format": int, "parse_tool_calls": bool}
//              (fields from rlc_chat_apply output)
// Returns the parsed message as OpenAI-format JSON (free with rlc_free):
//   {"role": "assistant", "content": str|null,
//    "tool_calls": [{"type": "function", "id": str,
//                    "function": {"name": str, "arguments": str}}], ...}
// On failure returns NULL and sets *err.
RLC_API char * rlc_chat_parse(const char * input, bool is_partial, const char * params_json, char ** err);

// params_json (all fields optional, llama.cpp defaults apply):
//   {"seed": int, "top_k": int, "top_p": f, "min_p": f, "typ_p": f, "temp": f,
//    "penalty_last_n": int, "penalty_repeat": f, "penalty_freq": f,
//    "penalty_present": f, "dry_multiplier": f, "dry_base": f,
//    "dry_allowed_length": int, "dry_penalty_last_n": int,
//    "xtc_probability": f, "xtc_threshold": f, "mirostat": int,
//    "mirostat_tau": f, "mirostat_eta": f, "min_keep": int,
//    "grammar": str, "grammar_type": int (1=user gbnf, 3=tool calls),
//    "grammar_lazy": bool,
//    "grammar_triggers": [{"type": int, "value": str, "token": int}],
//    "generation_prompt": str (prefilled into non-lazy tool-call grammars)}
// On failure returns NULL and sets *err.
RLC_API rlc_sampler * rlc_sampler_init(const llama_model * model, const char * params_json, char ** err);

RLC_API void rlc_sampler_free(rlc_sampler * smpl);

// Sample the next token from ctx logits at position idx (-1 = last).
// Applies the configured chain including grammar constraints.
RLC_API int32_t rlc_sampler_sample(rlc_sampler * smpl, llama_context * ctx, int32_t idx);

// Accept a token into sampler state (penalties, grammar advance).
RLC_API void rlc_sampler_accept(rlc_sampler * smpl, int32_t token, bool accept_grammar);

RLC_API void rlc_sampler_reset(rlc_sampler * smpl);

// Human-readable description of the sampler chain (free with rlc_free).
RLC_API char * rlc_sampler_print(rlc_sampler * smpl);

#ifdef __cplusplus
}
#endif

#endif
