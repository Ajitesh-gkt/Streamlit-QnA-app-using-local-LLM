# Prompt templates for the LLM model
memory_prompt_template = """<s>[INST] You are an LLM having a conversation with a human. Answer his questions.
    Previous conversation: {history}
    Human: {human_input}
    AI: [/INST]"""