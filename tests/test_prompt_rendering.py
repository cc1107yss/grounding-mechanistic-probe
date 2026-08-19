from __future__ import annotations

import unittest

from mechanistic_probe.extract import (
    answer_candidates,
    build_prompt,
    render_prompt,
)


class FakeTokenizer:
    chat_template = "official-template-placeholder"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.messages = messages
        text = "<|im_start|>system\nYou are Qwen.<|im_end|>\n"
        for message in messages:
            text += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text


def row(name: str, answer: bool = True):
    return {
        "example_id": name,
        "statements": [f"{name} fact one.", f"{name} fact two."],
        "question": f"{name} conclusion.",
        "answer": answer,
    }


class PromptRenderingTest(unittest.TestCase):
    def test_raw_prompt_is_unchanged(self):
        target = row("target")
        demos = [row("demo-true", True), row("demo-false", False)]
        prompt = render_prompt(FakeTokenizer(), target, demos, "raw")
        expected = (
            "demo-true fact one. demo-true fact two. demo-true conclusion. True or False? True\n"
            "demo-false fact one. demo-false fact two. demo-false conclusion. True or False? False\n"
            "target fact one. target fact two. target conclusion. True or False?"
        )
        self.assertEqual(prompt.text, expected)
        self.assertEqual(answer_candidates("raw"), (" True", " False"))

    def test_chat_multiturn_uses_native_roles_and_exact_target_spans(self):
        tokenizer = FakeTokenizer()
        target = row("target")
        demos = [row("demo-true", True), row("demo-false", False)]
        prompt = render_prompt(tokenizer, target, demos, "chat-multiturn")
        self.assertEqual([message["role"] for message in tokenizer.messages], ["user", "assistant", "user", "assistant", "user"])
        self.assertEqual(tokenizer.messages[1]["content"], "True")
        self.assertEqual(tokenizer.messages[3]["content"], "False")
        self.assertTrue(prompt.text.startswith("<|im_start|>system\nYou are Qwen."))
        self.assertTrue(prompt.text.endswith("<|im_start|>assistant\n"))
        for index, span in enumerate(prompt.statement_spans):
            self.assertEqual(prompt.text[slice(*span)], target["statements"][index])
        target_only = build_prompt(target, [])
        self.assertEqual(prompt.text[slice(*prompt.query_span)], target_only.text[slice(*target_only.query_span)])
        self.assertEqual(answer_candidates("chat-multiturn"), ("True", "False"))


if __name__ == "__main__":
    unittest.main()
