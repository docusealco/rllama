#include "llama_common.h"

#include "chat.h"
#include "common.h"
#include "sampling.h"

#include "nlohmann/json.hpp"

#include <cstring>
#include <string>

using json = nlohmann::ordered_json;

struct rlc_chat_templates {
    common_chat_templates_ptr ptr;
};

struct rlc_sampler {
    common_sampler * ptr;
};

static char * dup_string(const std::string & s) {
    char * out = (char *) malloc(s.size() + 1);
    memcpy(out, s.c_str(), s.size() + 1);
    return out;
}

static void set_err(char ** err, const std::string & msg) {
    if (err) {
        *err = dup_string(msg);
    }
}

void rlc_free(char * ptr) {
    free(ptr);
}

rlc_chat_templates * rlc_chat_templates_init(const llama_model * model,
                                             const char * template_src,
                                             const char * bos_token,
                                             const char * eos_token,
                                             char ** err) {
    try {
        auto tmpls = common_chat_templates_init(model, template_src, bos_token, eos_token);
        return new rlc_chat_templates{std::move(tmpls)};
    } catch (const std::exception & e) {
        set_err(err, e.what());
        return nullptr;
    }
}

void rlc_chat_templates_free(rlc_chat_templates * tmpls) {
    delete tmpls;
}

char * rlc_chat_apply(rlc_chat_templates * tmpls, const char * inputs_json, char ** err) {
    try {
        const json j = json::parse(inputs_json);

        common_chat_templates_inputs inputs;
        inputs.messages = common_chat_msgs_parse_oaicompat(j.at("messages"));

        if (j.contains("tools") && !j["tools"].is_null()) {
            inputs.tools = common_chat_tools_parse_oaicompat(j["tools"]);
        }

        inputs.add_generation_prompt = j.value("add_generation_prompt", true);
        inputs.enable_thinking       = j.value("enable_thinking", true);
        inputs.reasoning_format      = (common_reasoning_format) j.value("reasoning_format", (int) inputs.reasoning_format);
        inputs.parallel_tool_calls   = j.value("parallel_tool_calls", false);
        inputs.add_bos               = j.value("add_bos", false);
        inputs.add_eos               = j.value("add_eos", false);

        const auto params = common_chat_templates_apply(tmpls->ptr.get(), inputs);

        json triggers = json::array();
        for (const auto & t : params.grammar_triggers) {
            triggers.push_back({
                {"type",  (int) t.type},
                {"value", t.value},
                {"token", t.token},
            });
        }

        json out = {
            {"prompt",             params.prompt},
            {"format",             (int) params.format},
            {"parser",             params.parser},
            {"grammar",            params.grammar},
            {"grammar_lazy",       params.grammar_lazy},
            {"grammar_triggers",   triggers},
            {"generation_prompt",  params.generation_prompt},
            {"additional_stops",   params.additional_stops},
            {"thinking_start_tag", params.thinking_start_tag},
            {"thinking_end_tag",   params.thinking_end_tag},
        };

        return dup_string(out.dump());
    } catch (const std::exception & e) {
        set_err(err, e.what());
        return nullptr;
    }
}

char * rlc_chat_parse(const char * input, bool is_partial, const char * params_json, char ** err) {
    try {
        const json j = json::parse(params_json);

        common_chat_parser_params params;
        params.format            = (common_chat_format) j.value("format", (int) COMMON_CHAT_FORMAT_CONTENT_ONLY);
        params.reasoning_format  = (common_reasoning_format) j.value("reasoning_format", (int) params.reasoning_format);
        params.generation_prompt = j.value("generation_prompt", std::string());
        params.parse_tool_calls  = j.value("parse_tool_calls", true);

        const std::string parser = j.value("parser", std::string());
        if (!parser.empty()) {
            params.parser.load(parser);
        }

        const auto msg = common_chat_parse(input, is_partial, params);

        return dup_string(msg.to_json_oaicompat().dump());
    } catch (const std::exception & e) {
        set_err(err, e.what());
        return nullptr;
    }
}

rlc_sampler * rlc_sampler_init(const llama_model * model, const char * params_json, char ** err) {
    try {
        const json j = json::parse(params_json);

        common_params_sampling sp;

        sp.seed               = j.value("seed", sp.seed);
        sp.top_k              = j.value("top_k", sp.top_k);
        sp.top_p              = j.value("top_p", sp.top_p);
        sp.min_p              = j.value("min_p", sp.min_p);
        sp.typ_p              = j.value("typ_p", sp.typ_p);
        sp.temp               = j.value("temp", sp.temp);
        sp.penalty_last_n     = j.value("penalty_last_n", sp.penalty_last_n);
        sp.penalty_repeat     = j.value("penalty_repeat", sp.penalty_repeat);
        sp.penalty_freq       = j.value("penalty_freq", sp.penalty_freq);
        sp.penalty_present    = j.value("penalty_present", sp.penalty_present);
        sp.dry_multiplier     = j.value("dry_multiplier", sp.dry_multiplier);
        sp.dry_base           = j.value("dry_base", sp.dry_base);
        sp.dry_allowed_length = j.value("dry_allowed_length", sp.dry_allowed_length);
        sp.dry_penalty_last_n = j.value("dry_penalty_last_n", sp.dry_penalty_last_n);
        sp.xtc_probability    = j.value("xtc_probability", sp.xtc_probability);
        sp.xtc_threshold      = j.value("xtc_threshold", sp.xtc_threshold);
        sp.mirostat           = j.value("mirostat", sp.mirostat);
        sp.mirostat_tau       = j.value("mirostat_tau", sp.mirostat_tau);
        sp.mirostat_eta       = j.value("mirostat_eta", sp.mirostat_eta);
        sp.min_keep           = j.value("min_keep", sp.min_keep);
        sp.grammar_lazy       = j.value("grammar_lazy", sp.grammar_lazy);
        sp.generation_prompt  = j.value("generation_prompt", sp.generation_prompt);

        const std::string grammar_str = j.value("grammar", std::string());
        if (!grammar_str.empty()) {
            const auto grammar_type = (common_grammar_type) j.value("grammar_type", (int) COMMON_GRAMMAR_TYPE_USER);
            sp.grammar = common_grammar(grammar_type, grammar_str);
        }

        if (j.contains("grammar_triggers")) {
            for (const auto & t : j["grammar_triggers"]) {
                common_grammar_trigger trigger;
                trigger.type  = (common_grammar_trigger_type) t.value("type", 0);
                trigger.value = t.value("value", std::string());
                trigger.token = t.value("token", trigger.token);
                sp.grammar_triggers.push_back(trigger);
            }
        }

        common_sampler * smpl = common_sampler_init(model, sp);
        if (!smpl) {
            set_err(err, "failed to initialize sampler");
            return nullptr;
        }

        return new rlc_sampler{smpl};
    } catch (const std::exception & e) {
        set_err(err, e.what());
        return nullptr;
    }
}

void rlc_sampler_free(rlc_sampler * smpl) {
    if (smpl) {
        common_sampler_free(smpl->ptr);
        delete smpl;
    }
}

int32_t rlc_sampler_sample(rlc_sampler * smpl, llama_context * ctx, int32_t idx) {
    return common_sampler_sample(smpl->ptr, ctx, idx);
}

void rlc_sampler_accept(rlc_sampler * smpl, int32_t token, bool accept_grammar) {
    common_sampler_accept(smpl->ptr, token, accept_grammar);
}

void rlc_sampler_reset(rlc_sampler * smpl) {
    common_sampler_reset(smpl->ptr);
}

char * rlc_sampler_print(rlc_sampler * smpl) {
    return dup_string(common_sampler_print(smpl->ptr));
}
