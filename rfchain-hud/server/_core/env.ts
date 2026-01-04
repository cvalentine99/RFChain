export const ENV = {
  // Server configuration
  port: parseInt(process.env.PORT ?? "3007", 10),
  host: process.env.HOST ?? "0.0.0.0",
  
  // Auth configuration (OAuth is optional for self-hosted)
  appId: process.env.VITE_APP_ID ?? "",
  cookieSecret: process.env.JWT_SECRET ?? "change-me-in-production",
  oAuthServerUrl: process.env.OAUTH_SERVER_URL ?? "",
  ownerOpenId: process.env.OWNER_OPEN_ID ?? "",
  
  // Database
  databaseUrl: process.env.DATABASE_URL ?? "",
  
  // Environment
  isProduction: process.env.NODE_ENV === "production",
  
  // Forge API (optional - for cloud LLM)
  forgeApiUrl: process.env.BUILT_IN_FORGE_API_URL ?? "",
  forgeApiKey: process.env.BUILT_IN_FORGE_API_KEY ?? "",
  
  // Python configuration for signal analysis
  pythonPath: process.env.PYTHON_PATH ?? "python3",
  analysisScriptPath: process.env.ANALYSIS_SCRIPT_PATH ?? "",
  analysisOutputDir: process.env.ANALYSIS_OUTPUT_DIR ?? "",
  uploadDir: process.env.UPLOAD_DIR ?? "",
  
  // Local LLM (Ollama) configuration
  ollamaHost: process.env.OLLAMA_HOST ?? "",
  ollamaModel: process.env.OLLAMA_MODEL ?? "llama3",
  
  // External LLM APIs (optional)
  anthropicApiKey: process.env.ANTHROPIC_API_KEY ?? "",
  openaiApiKey: process.env.OPENAI_API_KEY ?? "",
  
  // Self-hosted mode detection
  get isSelfHosted(): boolean {
    return !this.oAuthServerUrl || !this.appId;
  },
};
