import { describe, it, expect } from "vitest";

describe("OpenAI API Key Validation", () => {
  it("should have OPENAI_API_KEY environment variable set", () => {
    const apiKey = process.env.OPENAI_API_KEY;
    expect(apiKey).toBeDefined();
    expect(apiKey).not.toBe("");
    expect(apiKey?.startsWith("sk-")).toBe(true);
  });

  it("should successfully call OpenAI API for embeddings", async () => {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error("OPENAI_API_KEY not set");
    }

    const response = await fetch("https://api.openai.com/v1/embeddings", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "text-embedding-3-small",
        input: "Test signal analysis embedding",
      }),
    });

    expect(response.ok).toBe(true);
    
    const data = await response.json();
    expect(data.data).toBeDefined();
    expect(data.data[0].embedding).toBeDefined();
    expect(Array.isArray(data.data[0].embedding)).toBe(true);
    expect(data.data[0].embedding.length).toBe(1536); // text-embedding-3-small dimension
  });

  it("should successfully call OpenAI API for chat completions", async () => {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error("OPENAI_API_KEY not set");
    }

    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [
          { role: "user", content: "Say 'test successful' in exactly two words" }
        ],
        max_tokens: 10,
      }),
    });

    expect(response.ok).toBe(true);
    
    const data = await response.json();
    expect(data.choices).toBeDefined();
    expect(data.choices[0].message.content).toBeDefined();
  });
});
