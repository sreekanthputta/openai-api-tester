# OpenAI Model Free Tier Rate Limits

This document summarizes the free tier rate limits for various OpenAI models.

| Model                     | Slug                          | RPM         | RPD   | TPM     |
| :------------------------ | :---------------------------- | :---------- | :---- | :------ |
| DALL·E 2                  | dall-e-2                      | 5 (img/min) | -     | -       |
| DALL·E 3                  | dall-e-3                      | 1 (img/min) | -     | -       |
| GPT-4o mini Audio         | gpt-4o-mini-audio-preview     | 3           | 200   | 40,000  |
| GPT-4o mini Search Preview| gpt-4o-mini-search-preview    | 3           | 200   | 40,000  |
| GPT-4o mini               | gpt-4o-mini                   | 3           | 200   | 40,000  |
| omni-moderation           | omni-moderation-latest        | 250         | 5,000 | 10,000  |
| text-embedding-3-large    | text-embedding-3-large        | 100         | 2,000 | 40,000  |
| text-embedding-3-small    | text-embedding-3-small        | 100         | 2,000 | 40,000  |
| text-embedding-ada-002    | text-embedding-ada-002        | 100         | 2,000 | 40,000  |
| TTS-1                     | tts-1                         | 3           | 200   | -       |
| Whisper                   | whisper-1                     | 3           | 200   | -       |

**Legend:**
*   **RPM:** Requests Per Minute (or Images Per Minute for DALL·E)
*   **RPD:** Requests Per Day
*   **TPM:** Tokens Per Minute
*   **-**: Not Applicable / Not Specified
