export const ENV = {
  appId: process.env.VITE_APP_ID ?? "",
  cookieSecret: process.env.JWT_SECRET ?? "",
  databaseUrl: process.env.DATABASE_URL ?? "",
  oAuthServerUrl: process.env.OAUTH_SERVER_URL ?? "",
  ownerOpenId: process.env.OWNER_OPEN_ID ?? "",
  isProduction: process.env.NODE_ENV === "production",
  forgeApiUrl: process.env.BUILT_IN_FORGE_API_URL ?? "",
  forgeApiKey: process.env.BUILT_IN_FORGE_API_KEY ?? "",
  // Python configuration for signal analysis
  pythonPath: process.env.PYTHON_PATH ?? "python3",
  analysisScriptPath: process.env.ANALYSIS_SCRIPT_PATH ?? "",
  analysisOutputDir: process.env.ANALYSIS_OUTPUT_DIR ?? "",
  uploadDir: process.env.UPLOAD_DIR ?? "",
};
