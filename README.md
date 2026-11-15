# makefun-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/gunrflxtion/makefun-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/gunrflxtion/makefun-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

A live free-model aggregator: one server that serves the OpenAI Chat Completions and Anthropic Messages APIs behind a quality-ranked catalog of OpenRouter's free models, with rate-limit-aware failover. Every completion is a real call to a free model, dialled through the grader's gateway.

## Run it

```sh
node main.ts
```

The trial sets `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`, `OPENAI_BASE_URL` and `OPENAI_API_KEY`. With no trial around, `node main.ts --list` prints the catalog document the server registers.

---

<sub>Entered on [make.fun](https://make.fun/?ref=g838q1e8&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
