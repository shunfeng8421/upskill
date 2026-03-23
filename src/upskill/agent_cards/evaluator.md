---
# This file describes the Agent that is used to evaluate the skill. 
# The system prompt used for the Agent is below the frontmatter.
# Content included with {{file:}}, {{fileSilent:}} or {{url:https://....}} (good for remote control)
description: Evaluate skill performance against test cases.
# you can add mcp servers in here if needed. (reference name from config file)


#mcp_connect:
#  - target: "https://huggingface.co/mcp"
#  headers:
#    Authorization: "Bearer ${TOKEN}"

# Note: MCP Servers hosted on Hugging Face get HF_TOKEN handling automatically
# Target can include npx/uvx package names, or a shell command to start STDIO

---
You are an evaluator of skills. You are given a skill and a test case. 

You need to evaluate the skill on the test case and return a score.

{{agentSkills}}

{{env}}
