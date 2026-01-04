import "dotenv/config";
import express from "express";
import { createServer } from "http";
import net from "net";
import { createExpressMiddleware } from "@trpc/server/adapters/express";
import { registerOAuthRoutes } from "./oauth";
import { appRouter } from "../routers";
import { createContext } from "./context";
import { serveStatic, setupVite } from "./vite";
import multer from "multer";
import { storagePut, storageGet } from "../storage";
import { spawn } from "child_process";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import { fileURLToPath } from "url";

// ESM compatibility for __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Clean UV environment variables to avoid Python version conflicts (Claude's Solution 1)
function cleanPythonEnvironment() {
  const cleanEnv = { ...process.env };
  
  // Remove UV-specific variables that cause Python version conflicts
  const uvVars = [
    'UV_PYTHON',
    'UV_PYTHON_PREFERENCE', 
    'VIRTUAL_ENV',
    'PYTHONPATH',
    'PYTHONHOME',
    'PYTHON_HOME'
  ];
  
  uvVars.forEach(varName => {
    delete cleanEnv[varName];
  });
  
  // Remove UV paths from PATH
  if (cleanEnv.PATH) {
    cleanEnv.PATH = cleanEnv.PATH
      .split(':')
      .filter(p => !p.includes('.local/share/uv'))
      .join(':');
  }
  
  return cleanEnv;
}

function isPortAvailable(port: number): Promise<boolean> {
  return new Promise(resolve => {
    const server = net.createServer();
    server.listen(port, () => {
      server.close(() => resolve(true));
    });
    server.on("error", () => resolve(false));
  });
}

async function findAvailablePort(startPort: number = 3000): Promise<number> {
  for (let port = startPort; port < startPort + 20; port++) {
    if (await isPortAvailable(port)) {
      return port;
    }
  }
  throw new Error(`No available port found starting from ${startPort}`);
}

async function startServer() {
  const app = express();
  const server = createServer(app);
  // Configure body parser with larger size limit for file uploads
  app.use(express.json({ limit: "50mb" }));
  app.use(express.urlencoded({ limit: "50mb", extended: true }));
  // OAuth callback under /api/oauth/callback
  registerOAuthRoutes(app);
  
  // Local storage directory for uploaded signals
  const uploadsDir = path.join(__dirname, "..", "..", "uploads");
  const analysisOutputDir = path.join(__dirname, "..", "..", "analysis_output");
  fs.mkdirSync(uploadsDir, { recursive: true });
  fs.mkdirSync(analysisOutputDir, { recursive: true });
  
  // Serve analysis output files (plots, metrics) as static files
  app.use("/analysis_output", express.static(analysisOutputDir));
  
  // File upload endpoint - saves locally
  const upload = multer({ 
    storage: multer.diskStorage({
      destination: (req, file, cb) => cb(null, uploadsDir),
      filename: (req, file, cb) => {
        const uniqueName = `${Date.now()}-${Math.random().toString(36).substring(7)}-${file.originalname}`;
        cb(null, uniqueName);
      }
    }),
    limits: { fileSize: 500 * 1024 * 1024 } // 500MB limit
  });
  
  app.post("/api/upload", upload.single("file"), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "No file provided" });
      }
      
      const localPath = req.file.path;
      const filename = req.file.filename;
      
      console.log("[Upload] File saved locally:", localPath);
      
      res.json({ 
        url: localPath,  // Return local file path
        key: filename,
        localPath: localPath
      });
    } catch (error) {
      console.error("Upload error:", error);
      res.status(500).json({ error: "Upload failed" });
    }
  });
  
  // Audio upload endpoint for voice transcription
  const audioDir = path.join(__dirname, "..", "..", "audio_uploads");
  fs.mkdirSync(audioDir, { recursive: true });
  
  const audioUpload = multer({ 
    storage: multer.diskStorage({
      destination: (req, file, cb) => cb(null, audioDir),
      filename: (req, file, cb) => {
        const ext = path.extname(file.originalname) || '.webm';
        const uniqueName = `${Date.now()}-${Math.random().toString(36).substring(7)}${ext}`;
        cb(null, uniqueName);
      }
    }),
    limits: { fileSize: 16 * 1024 * 1024 } // 16MB limit for audio
  });
  
  app.post("/api/audio/upload", audioUpload.single("audio"), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "No audio file provided" });
      }
      
      const localPath = req.file.path;
      const filename = req.file.filename;
      
      // Upload to S3 for Whisper API access
      const audioBuffer = fs.readFileSync(localPath);
      const s3Key = `audio/${filename}`;
      const { url } = await storagePut(s3Key, audioBuffer, req.file.mimetype || 'audio/webm');
      
      console.log("[Audio] Uploaded to S3:", url);
      
      res.json({ 
        url: url,
        localPath: localPath,
        filename: filename
      });
    } catch (error) {
      console.error("Audio upload error:", error);
      res.status(500).json({ error: "Audio upload failed" });
    }
  });
  
  // Analysis execution endpoint
  // Serve analysis output files statically
  app.use("/analysis_output", express.static(analysisOutputDir));
  
  app.post("/api/analyze", express.json(), async (req, res) => {
    try {
      const { localPath, uploadId, sampleRate, centerFreq, dataFormat, enableDigital, enableV3 } = req.body;
      
      console.log("[Analysis] Starting analysis for uploadId:", uploadId, "localPath:", localPath);
      
      if (!localPath || !uploadId) {
        return res.status(400).json({ error: "Missing localPath or uploadId" });
      }
      
      // Verify local file exists
      if (!fs.existsSync(localPath)) {
        return res.status(400).json({ error: `File not found: ${localPath}` });
      }
      
      // Create output directory for this analysis
      const outputDir = path.join(analysisOutputDir, String(uploadId));
      fs.mkdirSync(outputDir, { recursive: true });
      
      console.log("[Analysis] Input file:", localPath, "Output dir:", outputDir);
      
      // Build analysis command
      const scriptPath = path.join(__dirname, "..", "run_analysis.py");
      const args = [
        scriptPath,
        localPath,  // Use local file path directly
        "-o", outputDir,
        "-r", String(sampleRate || 1e6),
        "-c", String(centerFreq || 0),
        "-f", dataFormat || "complex64",
        "--json"
      ];
      
      if (enableDigital) args.push("--digital");
      if (enableV3) args.push("--v3");
      
      console.log("[Analysis] Script path:", scriptPath);
      console.log("[Analysis] Args:", args);
      
      // Execute analysis with clean Python environment
      const python = spawn("/usr/bin/python3.11", args, { 
        cwd: path.dirname(scriptPath),
        env: {
          ...cleanPythonEnvironment(),
          PYTHONDONTWRITEBYTECODE: "1",
          // Explicitly set Python paths to system locations
          PYTHONPATH: "/usr/lib/python3.11:/usr/local/lib/python3.11/dist-packages",
          PYTHONHOME: "/usr"
        }
      });
      
      let stdout = "";
      let stderr = "";
      
      python.stdout.on("data", (data) => { 
        stdout += data.toString(); 
        console.log("[Analysis] stdout:", data.toString().substring(0, 200));
      });
      python.stderr.on("data", (data) => { 
        stderr += data.toString(); 
        console.error("[Analysis] stderr:", data.toString());
      });
      
      python.on("error", (err) => {
        console.error("[Analysis] Spawn error:", err);
        return res.status(500).json({ error: `Failed to spawn Python: ${err.message}` });
      });
      
      python.on("close", async (code) => {
        try {
          console.log("[Analysis] Python exited with code:", code);
          if (code !== 0) {
            console.error("[Analysis] Analysis stderr:", stderr);
            return res.status(500).json({ error: "Analysis failed", stderr, stdout });
          }
          
          // Parse JSON output
          let result;
          try {
            result = JSON.parse(stdout);
          } catch (e) {
            console.error("Failed to parse analysis output:", stdout);
            return res.status(500).json({ error: "Failed to parse analysis output" });
          }
          
          if (!result.success) {
            return res.status(500).json({ error: result.error });
          }
          
          // Build local URLs for plot files (served via /analysis_output)
          const plotUrls: Record<string, string> = {};
          if (result.plot_files && Array.isArray(result.plot_files)) {
            for (const plotPath of result.plot_files) {
              const plotName = path.basename(plotPath);
              // Copy plot to output directory
              const destPath = path.join(outputDir, plotName);
              fs.copyFileSync(plotPath, destPath);
              plotUrls[plotName] = `/analysis_output/${uploadId}/${plotName}`;
            }
          }
          
          // Copy metrics JSON to output directory
          let metricsUrl = "";
          if (result.metrics_file) {
            const metricsName = path.basename(result.metrics_file);
            const destPath = path.join(outputDir, metricsName);
            fs.copyFileSync(result.metrics_file, destPath);
            metricsUrl = `/analysis_output/${uploadId}/${metricsName}`;
          }
          
          res.json({
            success: true,
            metrics: result.metrics,
            plotUrls,
            metricsUrl,
            timestamp: result.timestamp
          });
        } catch (processError) {
          console.error("Error processing results:", processError);
          res.status(500).json({ error: "Failed to process analysis results" });
        }
      });
      
      python.on("error", (err) => {
        console.error("Python process error:", err);
        res.status(500).json({ error: "Failed to start analysis process" });
      });
      
    } catch (error) {
      console.error("Analysis error:", error);
      res.status(500).json({ error: "Analysis failed" });
    }
  });
  
  // tRPC API
  app.use(
    "/api/trpc",
    createExpressMiddleware({
      router: appRouter,
      createContext,
    })
  );
  // development mode uses Vite, production mode uses static files
  if (process.env.NODE_ENV === "development") {
    await setupVite(app, server);
  } else {
    serveStatic(app);
  }

  const preferredPort = parseInt(process.env.PORT || "3000");
  const port = await findAvailablePort(preferredPort);

  if (port !== preferredPort) {
    console.log(`Port ${preferredPort} is busy, using port ${port} instead`);
  }

  server.listen(port, () => {
    console.log(`Server running on http://localhost:${port}/`);
  });
}

startServer().catch(console.error);
