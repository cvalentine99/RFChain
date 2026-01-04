import { COOKIE_NAME } from "@shared/const";
import { getSessionCookieOptions } from "./_core/cookies";
import { systemRouter } from "./_core/systemRouter";
import { publicProcedure, protectedProcedure, router } from "./_core/trpc";
import { z } from "zod";
import * as db from "./db";
import os from "os";
import { execSync } from "child_process";

export const appRouter = router({
    // if you need to use socket.io, read and register route in server/_core/index.ts, all api should start with '/api/' so that the gateway can route correctly
  system: systemRouter,
  auth: router({
    me: publicProcedure.query(opts => opts.ctx.user),
    logout: publicProcedure.mutation(({ ctx }) => {
      const cookieOptions = getSessionCookieOptions(ctx.req);
      ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 });
      return {
        success: true,
      } as const;
    }),
  }),

  // Analysis router
  analysis: router({
    getRecent: protectedProcedure
      .input(z.object({ limit: z.number().min(1).max(50).default(10) }))
      .query(async ({ ctx, input }) => {
        return db.getRecentAnalysesWithUploads(ctx.user.id, input.limit);
      }),
    
    getById: protectedProcedure
      .input(z.object({ id: z.number() }))
      .query(async ({ input }) => {
        return db.getAnalysisResult(input.id);
      }),
    
    getBySignalId: protectedProcedure
      .input(z.object({ signalId: z.number() }))
      .query(async ({ input }) => {
        return db.getAnalysisBySignalId(input.signalId);
      }),
    
    save: protectedProcedure
      .input(z.object({
        signalUploadId: z.number(),
        metricsUrl: z.string().optional(),
        plotUrls: z.record(z.string(), z.string()).optional(),
        metrics: z.any(),
      }))
      .mutation(async ({ ctx, input }) => {
        // Extract key metrics from the analysis result
        const analysisMetrics = input.metrics?.metrics || {};
        const forensicPipeline = input.metrics?.forensic_pipeline || {};
        const anomalies = input.metrics?.anomalies || {};
        
        // Parse DC offset if it's a string
        let dcReal = 0;
        let dcImag = 0;
        if (analysisMetrics.dc_offset) {
          const dcStr = String(analysisMetrics.dc_offset);
          const match = dcStr.match(/\(([^,]+),\s*([^)]+)\)/);
          if (match) {
            dcReal = parseFloat(match[1]) || 0;
            dcImag = parseFloat(match[2]) || 0;
          }
        }
        
        // Create analysis result record - include plotUrls in fullMetrics
        const fullMetricsWithPlots = {
          ...input.metrics,
          plot_urls: input.plotUrls || {},
        };
        
        const analysisId = await db.createAnalysisResult({
          signalId: input.signalUploadId,
          userId: ctx.user.id,
          avgPowerDbm: analysisMetrics.avg_power_dbm || 0,
          peakPowerDbm: analysisMetrics.peak_power_dbm || 0,
          paprDb: analysisMetrics.papr_db || 0,
          iqImbalanceDb: analysisMetrics.iq_imbalance_db || 0,
          snrEstimateDb: analysisMetrics.snr_estimate_db || 0,
          bandwidthHz: analysisMetrics.bandwidth_estimate_hz || 0,
          freqOffsetHz: analysisMetrics.center_freq_offset_hz || 0,
          dcOffsetReal: dcReal,
          dcOffsetImag: dcImag,
          sampleCount: analysisMetrics.sample_count || 0,
          anomalies: anomalies,
          fullMetrics: fullMetricsWithPlots,
        });
        
        if (!analysisId) {
          throw new Error("Failed to create analysis result");
        }
        
        // Create forensic report if hash chain data exists
        if (forensicPipeline.hash_chain && forensicPipeline.hash_chain.length > 0) {
          const hashChain = forensicPipeline.hash_chain;
          const getHash = (stage: string, algo: 'sha256' | 'sha3_256' = 'sha256') => {
            const checkpoint = hashChain.find((c: any) => c.stage === stage);
            return checkpoint?.hashes?.[algo] || null;
          };
          
          await db.createForensicReport({
            analysisId: analysisId,
            userId: ctx.user.id,
            rawInputHash: getHash("raw_input", "sha256"),
            postMetricsHash: getHash("post_metrics", "sha256"),
            postAnomalyHash: getHash("post_anomaly", "sha256"),
            postDigitalHash: getHash("post_digital_analysis", "sha256"),
            postV3Hash: getHash("post_v3_analysis", "sha256"),
            preOutputHash: getHash("pre_output", "sha256"),
            rawInputHashSha3: getHash("raw_input", "sha3_256"),
            postMetricsHashSha3: getHash("post_metrics", "sha3_256"),
            postAnomalyHashSha3: getHash("post_anomaly", "sha3_256"),
            postDigitalHashSha3: getHash("post_digital_analysis", "sha3_256"),
            postV3HashSha3: getHash("post_v3_analysis", "sha3_256"),
            preOutputHashSha3: getHash("pre_output", "sha3_256"),
            forensicPipeline: forensicPipeline,
          });
        }
        
        // Update signal upload status
        await db.updateSignalUploadStatus(input.signalUploadId, "completed");
        
        // Auto-generate embedding for RAG (async, don't block response)
        (async () => {
          try {
            const { generateEmbedding, formatAnalysisForEmbedding, isEmbeddingError } = await import("./_core/embeddings");
            const upload = await db.getSignalUpload(input.signalUploadId);
            
            const contentText = formatAnalysisForEmbedding({
              filename: upload?.originalName,
              avgPowerDbm: analysisMetrics.avg_power_dbm,
              peakPowerDbm: analysisMetrics.peak_power_dbm,
              paprDb: analysisMetrics.papr_db,
              bandwidthHz: analysisMetrics.bandwidth_estimate_hz,
              freqOffsetHz: analysisMetrics.center_freq_offset_hz,
              iqImbalanceDb: analysisMetrics.iq_imbalance_db,
              snrEstimateDb: analysisMetrics.snr_estimate_db,
              sampleCount: analysisMetrics.sample_count,
              anomalies: anomalies,
              fullMetrics: fullMetricsWithPlots,
            });
            
            const embeddingResult = await generateEmbedding(contentText);
            
            if (!isEmbeddingError(embeddingResult)) {
              const hasAnomalies = anomalies ? Object.values(anomalies).some(v => v === true) : false;
              
              await db.upsertAnalysisEmbedding({
                analysisId: analysisId,
                userId: ctx.user.id,
                contentText,
                embedding: embeddingResult.embedding,
                signalFilename: upload?.originalName || null,
                avgPowerDbm: analysisMetrics.avg_power_dbm,
                bandwidthHz: analysisMetrics.bandwidth_estimate_hz,
                hasAnomalies: hasAnomalies ? 1 : 0,
              });
              console.log(`[RAG] Auto-generated embedding for analysis ${analysisId}`);
            }
          } catch (err) {
            console.warn(`[RAG] Failed to auto-generate embedding for analysis ${analysisId}:`, err);
          }
        })();
        
        return { analysisId };
      }),
  }),
  
  // Signal upload router
  signal: router({
    getUploads: protectedProcedure
      .input(z.object({ limit: z.number().min(1).max(50).default(20) }))
      .query(async ({ ctx, input }) => {
        return db.getUserSignalUploads(ctx.user.id, input.limit);
      }),
    
    getUpload: protectedProcedure
      .input(z.object({ id: z.number() }))
      .query(async ({ input }) => {
        return db.getSignalUpload(input.id);
      }),
    
    // Create upload record and get presigned URL
    createUpload: protectedProcedure
      .input(z.object({
        filename: z.string(),
        originalName: z.string(),
        fileSize: z.number(),
        mimeType: z.string().optional(),
      }))
      .mutation(async ({ ctx, input }) => {
        const { storagePut } = await import("./storage");
        
        // Generate unique S3 key
        const timestamp = Date.now();
        const randomSuffix = Math.random().toString(36).substring(2, 10);
        const s3Key = `signals/${ctx.user.id}/${timestamp}-${randomSuffix}-${input.filename}`;
        
        // Create database record
        const uploadId = await db.createSignalUpload({
          userId: ctx.user.id,
          filename: input.filename,
          originalName: input.originalName,
          fileSize: input.fileSize,
          mimeType: input.mimeType || "application/octet-stream",
          s3Key,
          s3Url: "", // Will be updated after upload
          status: "pending",
        });
        
        return {
          uploadId,
          s3Key,
        };
      }),
    
    // Complete upload after file is uploaded to S3
    completeUpload: protectedProcedure
      .input(z.object({
        uploadId: z.number(),
        s3Url: z.string(),
      }))
      .mutation(async ({ input }) => {
        const { getDb } = await import("./db");
        const { signalUploads } = await import("../drizzle/schema");
        const { eq } = await import("drizzle-orm");
        
        const dbInstance = await getDb();
        if (!dbInstance) throw new Error("Database not available");
        
        await dbInstance.update(signalUploads)
          .set({ s3Url: input.s3Url, status: "processing" })
          .where(eq(signalUploads.id, input.uploadId));
        
        return { success: true };
      }),
    
    // Update upload status
    updateStatus: protectedProcedure
      .input(z.object({
        uploadId: z.number(),
        status: z.enum(["pending", "processing", "completed", "failed"]),
      }))
      .mutation(async ({ input }) => {
        await db.updateSignalUploadStatus(input.uploadId, input.status);
        return { success: true };
      }),
  }),
  
  // Forensic reports router
  forensic: router({
    getByAnalysisId: protectedProcedure
      .input(z.object({ analysisId: z.number() }))
      .query(async ({ input }) => {
        return db.getForensicReport(input.analysisId);
      }),
    
    getUserReports: protectedProcedure
      .input(z.object({ limit: z.number().min(1).max(50).default(20) }))
      .query(async ({ ctx, input }) => {
        return db.getUserForensicReports(ctx.user.id, input.limit);
      }),
    
    // Generate PDF report
    generatePdf: protectedProcedure
      .input(z.object({ analysisId: z.number() }))
      .mutation(async ({ ctx, input }) => {
        const { generateForensicPDF } = await import("./_core/pdfGenerator");
        
        // Get analysis data
        const analysis = await db.getAnalysisResult(input.analysisId);
        if (!analysis) {
          throw new Error("Analysis not found");
        }
        
        // Check ownership
        if (analysis.userId !== ctx.user.id) {
          throw new Error("Unauthorized");
        }
        
        // Get signal upload info
        const upload = await db.getSignalUpload(analysis.signalId);
        
        // Get forensic report
        const forensicReport = await db.getForensicReport(input.analysisId);
        
        // Build hash chain from forensic report
        const hashChain: { stage: string; sha256: string | null; sha3_256: string | null }[] = [];
        if (forensicReport) {
          const stages = [
            { stage: "raw_input", sha256: forensicReport.rawInputHash, sha3_256: forensicReport.rawInputHashSha3 },
            { stage: "post_metrics", sha256: forensicReport.postMetricsHash, sha3_256: forensicReport.postMetricsHashSha3 },
            { stage: "post_anomaly", sha256: forensicReport.postAnomalyHash, sha3_256: forensicReport.postAnomalyHashSha3 },
            { stage: "post_digital_analysis", sha256: forensicReport.postDigitalHash, sha3_256: forensicReport.postDigitalHashSha3 },
            { stage: "post_v3_analysis", sha256: forensicReport.postV3Hash, sha3_256: forensicReport.postV3HashSha3 },
            { stage: "pre_output", sha256: forensicReport.preOutputHash, sha3_256: forensicReport.preOutputHashSha3 },
          ];
          // Only include stages that have at least one hash
          stages.forEach(s => {
            if (s.sha256 || s.sha3_256) {
              hashChain.push(s);
            }
          });
        }
        
        // Parse anomalies
        const anomalies = (analysis.anomalies as Record<string, boolean>) || {};
        
        // Parse full metrics for config
        const fullMetrics = (analysis.fullMetrics as Record<string, unknown>) || {};
        const analysisConfig = (fullMetrics.analysis_config as Record<string, unknown>) || {};
        
        // Generate PDF
        const pdfBuffer = await generateForensicPDF({
          analysisId: input.analysisId,
          signalFilename: upload?.originalName || "Unknown",
          analysisDate: analysis.createdAt,
          metrics: {
            avgPowerDbm: analysis.avgPowerDbm ?? 0,
            peakPowerDbm: analysis.peakPowerDbm ?? 0,
            paprDb: analysis.paprDb ?? 0,
            bandwidthHz: analysis.bandwidthHz ?? 0,
            freqOffsetHz: analysis.freqOffsetHz ?? 0,
            iqImbalanceDb: analysis.iqImbalanceDb ?? 0,
            snrEstimateDb: analysis.snrEstimateDb ?? 0,
            sampleCount: analysis.sampleCount ?? 0,
            dcOffsetReal: analysis.dcOffsetReal ?? 0,
            dcOffsetImag: analysis.dcOffsetImag ?? 0,
          },
          config: {
            sampleRate: Number(analysisConfig.sample_rate) || 1000000,
            centerFreq: Number(analysisConfig.center_freq) || 0,
            dataFormat: String(analysisConfig.data_format || "complex64"),
            fftSize: Number(analysisConfig.fft_size) || 4096,
            analyzerVersion: String(analysisConfig.analyzer_version || "2.2.2-forensic"),
          },
          anomalies: {
            dcSpike: anomalies.dc_spike || false,
            saturation: anomalies.saturation || false,
            dropout: anomalies.dropout || false,
            ...anomalies,
          },
          hashChain,
          analyst: {
            name: ctx.user.name || "Unknown",
            id: ctx.user.openId,
          },
        });
        
        // Return as base64
        return {
          pdf: pdfBuffer.toString("base64"),
          filename: `forensic_report_${input.analysisId}.pdf`,
        };
      }),
  }),
  
  // Chat router with RAG
  chat: router({
    send: protectedProcedure
      .input(z.object({
        message: z.string(),
        context: z.record(z.string(), z.unknown()).optional(),
        sessionId: z.string().optional(),
        useRag: z.boolean().default(true),
      }))
      .mutation(async ({ ctx, input }) => {
        const { invokeLLM } = await import("./_core/llm");
        const { generateEmbedding, isEmbeddingError } = await import("./_core/embeddings");
        
        // Build system prompt with RAG context
        let systemPrompt = `You are JARVIS, an advanced AI assistant specialized in RF signal analysis and forensic investigation. You speak in a professional, slightly formal British manner, similar to the AI from Iron Man.

Your capabilities include:
- Analyzing RF signal characteristics (power levels, frequency, bandwidth, modulation)
- Interpreting forensic hash chain verification results
- Explaining signal anomalies and their potential causes
- Providing insights on signal quality metrics (SNR, PAPR, I/Q imbalance)
- Guiding users through the forensic analysis process

Always be helpful, precise, and maintain a calm, professional demeanor. When discussing technical matters, provide clear explanations that balance accuracy with accessibility.`;
        
        // RAG: Find relevant past analyses
        let ragContext = "";
        if (input.useRag) {
          try {
            // Generate embedding for the user's query
            const queryEmbedding = await generateEmbedding(input.message);
            
            if (!isEmbeddingError(queryEmbedding)) {
              // Find similar analyses
              const similarAnalyses = await db.findSimilarAnalyses(
                ctx.user.id,
                queryEmbedding.embedding,
                3, // Top 3 most relevant
                0.4 // Minimum similarity threshold
              );
              
              if (similarAnalyses.length > 0) {
                ragContext = "\n\nRelevant Past Analyses (from your history):\n";
                similarAnalyses.forEach((analysis, idx) => {
                  ragContext += `\n--- Analysis ${idx + 1} (${Math.round(analysis.similarity * 100)}% relevant) ---\n`;
                  ragContext += `File: ${analysis.signalFilename || "Unknown"}\n`;
                  ragContext += analysis.contentText + "\n";
                });
              }
            }
          } catch (ragError) {
            console.warn("RAG retrieval failed, continuing without context:", ragError);
          }
        }
        
        // Add current context if provided
        if (input.context) {
          systemPrompt += `\n\nCurrent Analysis Context:\n${JSON.stringify(input.context, null, 2)}`;
        }
        
        // Add RAG context
        if (ragContext) {
          systemPrompt += ragContext;
        }
        
        try {
          const response = await invokeLLM({
            messages: [
              { role: "system", content: systemPrompt },
              { role: "user", content: input.message },
            ],
          });
          
          const rawContent = response.choices[0]?.message?.content;
          const assistantMessage = typeof rawContent === "string" 
            ? rawContent 
            : Array.isArray(rawContent) 
              ? rawContent.map(c => c.type === "text" ? c.text : "").join("")
              : "I apologize, but I was unable to process your request.";
          
          // Save to chat history
          await db.createChatMessage({
            userId: ctx.user.id,
            sessionId: input.sessionId || `session-${Date.now()}`,
            role: "user",
            content: input.message,
            analysisContext: input.context,
          });
          
          await db.createChatMessage({
            userId: ctx.user.id,
            sessionId: input.sessionId || `session-${Date.now()}`,
            role: "assistant",
            content: assistantMessage,
          });
          
          return { response: assistantMessage, ragUsed: ragContext.length > 0 };
        } catch (error) {
          console.error("LLM error:", error);
          return { response: "I apologize, but I encountered an error processing your request. Please try again.", ragUsed: false };
        }
      }),
    
    getHistory: protectedProcedure
      .input(z.object({ sessionId: z.string(), limit: z.number().default(50) }))
      .query(async ({ input }) => {
        return db.getSessionMessages(input.sessionId, input.limit);
      }),
  }),
  
  // Voice transcription router
  voice: router({
    transcribe: protectedProcedure
      .input(z.object({
        audioUrl: z.string(),
        language: z.string().optional(),
        prompt: z.string().optional(),
      }))
      .mutation(async ({ input }) => {
        const { transcribeAudio } = await import("./_core/voiceTranscription");
        
        const result = await transcribeAudio({
          audioUrl: input.audioUrl,
          language: input.language,
          prompt: input.prompt || "Transcribe the user's voice command for RF signal analysis",
        });
        
        // Check if it's an error
        if ('error' in result) {
          throw new Error(result.error);
        }
        
        return {
          text: result.text,
          language: result.language,
          duration: result.duration,
        };
      }),
  }),
  
  // Embedding router for RAG
  embedding: router({
    // Generate embedding for a specific analysis
    generateForAnalysis: protectedProcedure
      .input(z.object({ analysisId: z.number() }))
      .mutation(async ({ ctx, input }) => {
        const { generateEmbedding, formatAnalysisForEmbedding, isEmbeddingError } = await import("./_core/embeddings");
        
        // Get the analysis
        const analysis = await db.getAnalysisResult(input.analysisId);
        if (!analysis) {
          throw new Error("Analysis not found");
        }
        
        // Check if user owns this analysis
        if (analysis.userId !== ctx.user.id) {
          throw new Error("Unauthorized");
        }
        
        // Get the signal upload for filename
        const upload = await db.getSignalUpload(analysis.signalId);
        
        // Format the analysis data for embedding
        const contentText = formatAnalysisForEmbedding({
          filename: upload?.originalName,
          avgPowerDbm: analysis.avgPowerDbm,
          peakPowerDbm: analysis.peakPowerDbm,
          paprDb: analysis.paprDb,
          bandwidthHz: analysis.bandwidthHz,
          freqOffsetHz: analysis.freqOffsetHz,
          iqImbalanceDb: analysis.iqImbalanceDb,
          snrEstimateDb: analysis.snrEstimateDb,
          sampleCount: analysis.sampleCount,
          anomalies: analysis.anomalies as Record<string, unknown> | null,
          fullMetrics: analysis.fullMetrics as Record<string, unknown> | null,
        });
        
        // Generate embedding
        const embeddingResult = await generateEmbedding(contentText);
        
        if (isEmbeddingError(embeddingResult)) {
          throw new Error(`Embedding generation failed: ${embeddingResult.error}`);
        }
        
        // Check if anomalies exist
        const anomalies = analysis.anomalies as Record<string, boolean> | null;
        const hasAnomalies = anomalies ? Object.values(anomalies).some(v => v === true) : false;
        
        // Save to database
        await db.upsertAnalysisEmbedding({
          analysisId: input.analysisId,
          userId: ctx.user.id,
          contentText,
          embedding: embeddingResult.embedding,
          signalFilename: upload?.originalName || null,
          analysisDate: analysis.createdAt,
          avgPowerDbm: analysis.avgPowerDbm,
          bandwidthHz: analysis.bandwidthHz,
          hasAnomalies: hasAnomalies ? 1 : 0,
        });
        
        return { success: true, tokensUsed: embeddingResult.usage.total_tokens };
      }),
    
    // Backfill embeddings for all analyses without them
    backfill: protectedProcedure
      .input(z.object({ limit: z.number().min(1).max(100).default(50) }))
      .mutation(async ({ ctx, input }) => {
        const { generateEmbedding, formatAnalysisForEmbedding, isEmbeddingError } = await import("./_core/embeddings");
        
        // Get analyses without embeddings
        const analysesToEmbed = await db.getAnalysesWithoutEmbeddings(ctx.user.id, input.limit);
        
        let successCount = 0;
        let errorCount = 0;
        const errors: string[] = [];
        
        for (const analysis of analysesToEmbed) {
          try {
            const upload = await db.getSignalUpload(analysis.signalId);
            
            const contentText = formatAnalysisForEmbedding({
              filename: upload?.originalName,
              avgPowerDbm: analysis.avgPowerDbm,
              peakPowerDbm: analysis.peakPowerDbm,
              paprDb: analysis.paprDb,
              bandwidthHz: analysis.bandwidthHz,
              freqOffsetHz: analysis.freqOffsetHz,
              iqImbalanceDb: analysis.iqImbalanceDb,
              snrEstimateDb: analysis.snrEstimateDb,
              sampleCount: analysis.sampleCount,
              anomalies: analysis.anomalies as Record<string, unknown> | null,
              fullMetrics: analysis.fullMetrics as Record<string, unknown> | null,
            });
            
            const embeddingResult = await generateEmbedding(contentText);
            
            if (isEmbeddingError(embeddingResult)) {
              errorCount++;
              errors.push(`Analysis ${analysis.id}: ${embeddingResult.error}`);
              continue;
            }
            
            const anomalies = analysis.anomalies as Record<string, boolean> | null;
            const hasAnomalies = anomalies ? Object.values(anomalies).some(v => v === true) : false;
            
            await db.upsertAnalysisEmbedding({
              analysisId: analysis.id,
              userId: ctx.user.id,
              contentText,
              embedding: embeddingResult.embedding,
              signalFilename: upload?.originalName || null,
              analysisDate: analysis.createdAt,
              avgPowerDbm: analysis.avgPowerDbm,
              bandwidthHz: analysis.bandwidthHz,
              hasAnomalies: hasAnomalies ? 1 : 0,
            });
            
            successCount++;
          } catch (err) {
            errorCount++;
            errors.push(`Analysis ${analysis.id}: ${err instanceof Error ? err.message : "Unknown error"}`);
          }
        }
        
        return {
          total: analysesToEmbed.length,
          success: successCount,
          errors: errorCount,
          errorDetails: errors.slice(0, 10),
        };
      }),
    
    // Get embedding status
    getStatus: protectedProcedure.query(async ({ ctx }) => {
      const allEmbeddings = await db.getAllUserEmbeddings(ctx.user.id);
      const analysesWithoutEmbeddings = await db.getAnalysesWithoutEmbeddings(ctx.user.id, 100);
      
      return {
        embeddedCount: allEmbeddings.length,
        pendingCount: analysesWithoutEmbeddings.length,
        isComplete: analysesWithoutEmbeddings.length === 0,
      };
    }),
  }),
  
  // LLM config router
  llmConfig: router({
    get: protectedProcedure.query(async ({ ctx }) => {
      return db.getUserLlmConfig(ctx.user.id);
    }),
    
    update: protectedProcedure
      .input(z.object({
        provider: z.enum(["builtin", "openai", "anthropic", "local"]),
        apiKey: z.string().optional(),
        localEndpoint: z.string().optional(),
        model: z.string().optional(),
      }))
      .mutation(async ({ ctx, input }) => {
        await db.upsertLlmConfig({
          userId: ctx.user.id,
          ...input,
        });
        return { success: true };
      }),
  }),
  
  // RAG settings router
  ragSettings: router({
    get: protectedProcedure.query(async ({ ctx }) => {
      const settings = await db.getRagSettings(ctx.user.id);
      const stats = await db.getEmbeddingStats(ctx.user.id);
      
      return {
        settings: settings || {
          enabled: 1,
          similarityThreshold: 0.7,
          maxResults: 5,
          autoEmbed: 1,
        },
        stats,
      };
    }),
    
    update: protectedProcedure
      .input(z.object({
        enabled: z.number().min(0).max(1).optional(),
        similarityThreshold: z.number().min(0).max(1).optional(),
        maxResults: z.number().min(1).max(20).optional(),
        autoEmbed: z.number().min(0).max(1).optional(),
      }))
      .mutation(async ({ ctx, input }) => {
        await db.upsertRagSettings(ctx.user.id, input);
        return { success: true };
      }),
  }),

  // Batch processing router
  batch: router({
    // Create a new batch job
    create: protectedProcedure
      .input(z.object({
        name: z.string().optional(),
        files: z.array(z.object({
          filename: z.string(),
          localPath: z.string(),
        })),
        options: z.object({
          sampleRate: z.number().default(1000000),
          dataFormat: z.string().default('complex64'),
          digitalAnalysis: z.boolean().default(false),
          v3Analysis: z.boolean().default(false),
        }).optional(),
      }))
      .mutation(async ({ ctx, input }) => {
        // Create batch job
        const jobId = await db.createBatchJob({
          userId: ctx.user.id,
          name: input.name || `Batch ${new Date().toISOString()}`,
          totalFiles: input.files.length,
          completedFiles: 0,
          failedFiles: 0,
          status: 'pending',
          sampleRate: input.options?.sampleRate || 1000000,
          dataFormat: input.options?.dataFormat || 'complex64',
          digitalAnalysis: input.options?.digitalAnalysis ? 1 : 0,
          v3Analysis: input.options?.v3Analysis ? 1 : 0,
        });
        
        if (!jobId) {
          throw new Error('Failed to create batch job');
        }
        
        // Create queue items for each file
        for (let i = 0; i < input.files.length; i++) {
          await db.createBatchQueueItem({
            batchJobId: jobId,
            filename: input.files[i].filename,
            localPath: input.files[i].localPath,
            status: 'queued',
            position: i + 1,
          });
        }
        
        return { jobId, totalFiles: input.files.length };
      }),
    
    // Get batch job status
    getJob: protectedProcedure
      .input(z.object({ jobId: z.number() }))
      .query(async ({ ctx, input }) => {
        const job = await db.getBatchJob(input.jobId);
        if (!job || job.userId !== ctx.user.id) {
          throw new Error('Batch job not found');
        }
        
        const queueItems = await db.getBatchQueueItems(input.jobId);
        
        return {
          ...job,
          queueItems,
          progress: job.totalFiles > 0 
            ? Math.round(((job.completedFiles + job.failedFiles) / job.totalFiles) * 100)
            : 0,
        };
      }),
    
    // Get user's batch jobs
    getJobs: protectedProcedure
      .input(z.object({ limit: z.number().min(1).max(50).default(10) }))
      .query(async ({ ctx, input }) => {
        return db.getUserBatchJobs(ctx.user.id, input.limit);
      }),
    
    // Start processing a batch job
    start: protectedProcedure
      .input(z.object({ jobId: z.number() }))
      .mutation(async ({ ctx, input }) => {
        const job = await db.getBatchJob(input.jobId);
        if (!job || job.userId !== ctx.user.id) {
          throw new Error('Batch job not found');
        }
        
        if (job.status !== 'pending') {
          throw new Error('Batch job is not in pending state');
        }
        
        // Update job status to processing
        await db.updateBatchJobStatus(input.jobId, 'processing', {
          startedAt: new Date(),
        });
        
        return { success: true, message: 'Batch processing started' };
      }),
    
    // Get next item to process
    getNextItem: protectedProcedure
      .input(z.object({ jobId: z.number() }))
      .query(async ({ ctx, input }) => {
        const job = await db.getBatchJob(input.jobId);
        if (!job || job.userId !== ctx.user.id) {
          throw new Error('Batch job not found');
        }
        
        return db.getNextQueueItem(input.jobId);
      }),
    
    // Update queue item status
    updateItem: protectedProcedure
      .input(z.object({
        itemId: z.number(),
        status: z.enum(['queued', 'processing', 'completed', 'failed']),
        signalUploadId: z.number().optional(),
        analysisResultId: z.number().optional(),
        errorMessage: z.string().optional(),
      }))
      .mutation(async ({ ctx, input }) => {
        const updates: any = {};
        if (input.signalUploadId) updates.signalUploadId = input.signalUploadId;
        if (input.analysisResultId) updates.analysisResultId = input.analysisResultId;
        if (input.errorMessage) updates.errorMessage = input.errorMessage;
        if (input.status === 'processing') updates.startedAt = new Date();
        if (input.status === 'completed' || input.status === 'failed') updates.completedAt = new Date();
        
        await db.updateQueueItemStatus(input.itemId, input.status, updates);
        return { success: true };
      }),
    
    // Increment job progress
    incrementProgress: protectedProcedure
      .input(z.object({ jobId: z.number(), success: z.boolean() }))
      .mutation(async ({ ctx, input }) => {
        await db.incrementBatchJobProgress(input.jobId, input.success);
        
        // Check if job is complete
        const job = await db.getBatchJob(input.jobId);
        if (job && (job.completedFiles + job.failedFiles) >= job.totalFiles) {
          await db.updateBatchJobStatus(input.jobId, 'completed', {
            completedAt: new Date(),
          });
        }
        
        return { success: true };
      }),
    
    // Cancel a batch job
    cancel: protectedProcedure
      .input(z.object({ jobId: z.number() }))
      .mutation(async ({ ctx, input }) => {
        const job = await db.getBatchJob(input.jobId);
        if (!job || job.userId !== ctx.user.id) {
          throw new Error('Batch job not found');
        }
        
        await db.updateBatchJobStatus(input.jobId, 'cancelled', {
          completedAt: new Date(),
        });
        
        return { success: true };
      }),
    
    // Get active batch job for user
    getActive: protectedProcedure
      .query(async ({ ctx }) => {
        return db.getActiveBatchJob(ctx.user.id);
      }),
  }),
  
  // GPU monitoring router
  gpu: router({
    // Get current GPU stats
    getStats: publicProcedure.query(async () => {
      const { spawn } = await import('child_process');
      const path = await import('path');
      const { fileURLToPath } = await import('url');
      
      const __filename = fileURLToPath(import.meta.url);
      const __dirname = path.dirname(__filename);
      const scriptPath = path.join(__dirname, 'gpu_monitor.py');
      
      return new Promise((resolve) => {
        const python = spawn('python3', [scriptPath]);
        let stdout = '';
        let stderr = '';
        
        python.stdout.on('data', (data) => { stdout += data.toString(); });
        python.stderr.on('data', (data) => { stderr += data.toString(); });
        
        python.on('close', (code) => {
          if (code === 0) {
            try {
              const result = JSON.parse(stdout);
              resolve(result);
            } catch (e) {
              resolve({ success: false, error: 'Failed to parse GPU stats', gpus: [] });
            }
          } else {
            resolve({ success: false, error: stderr || 'GPU monitoring failed', gpus: [] });
          }
        });
        
        python.on('error', () => {
          resolve({ success: false, error: 'Failed to run GPU monitor', gpus: [] });
        });
        
        // Timeout after 5 seconds
        setTimeout(() => {
          python.kill();
          resolve({ success: false, error: 'GPU monitoring timed out', gpus: [] });
        }, 5000);
      });
    }),
    
    // Run GPU benchmark and save to history
    runBenchmark: protectedProcedure.mutation(async ({ ctx }) => {
      const { spawn } = await import('child_process');
      const path = await import('path');
      const { fileURLToPath } = await import('url');
      
      const __filename = fileURLToPath(import.meta.url);
      const __dirname = path.dirname(__filename);
      const scriptPath = path.join(__dirname, 'gpu_benchmark.py');
      
      const result: any = await new Promise((resolve) => {
        const python = spawn('python3', [scriptPath]);
        let stdout = '';
        let stderr = '';
        
        python.stdout.on('data', (data) => { stdout += data.toString(); });
        python.stderr.on('data', (data) => { stderr += data.toString(); });
        
        python.on('close', (code) => {
          if (code === 0) {
            try {
              const result = JSON.parse(stdout);
              resolve(result);
            } catch (e) {
              resolve({ success: false, error: 'Failed to parse benchmark results' });
            }
          } else {
            resolve({ success: false, error: stderr || 'Benchmark failed' });
          }
        });
        
        python.on('error', () => {
          resolve({ success: false, error: 'Failed to run benchmark' });
        });
        
        // Timeout after 60 seconds for benchmark
        setTimeout(() => {
          python.kill();
          resolve({ success: false, error: 'Benchmark timed out' });
        }, 60000);
      });
      
      // Save benchmark result to database
      try {
        const historyId = await db.saveBenchmarkResult({
          userId: ctx.user.id,
          gpuName: result.gpu_info?.name || null,
          gpuMemoryMb: result.gpu_info?.total_memory_gb ? Math.round(result.gpu_info.total_memory_gb * 1024) : null,
          cudaVersion: null,
          driverVersion: null,
          avgSpeedup: result.summary?.avg_speedup || null,
          maxSpeedup: result.summary?.max_speedup || null,
          minSpeedup: result.summary?.min_speedup || null,
          benchmarkResults: result.benchmarks || [],
          systemInfo: {
            gpu_available: result.gpu_available,
            timestamp: result.timestamp,
            gpu_info: result.gpu_info,
          },
          success: result.success ? 1 : 0,
          errorMessage: result.error || null,
        });
        
        return { ...result, historyId };
      } catch (e) {
        // Return benchmark result even if saving fails
        console.error('Failed to save benchmark history:', e);
        return result;
      }
    }),
    
    // Get benchmark history
    getHistory: protectedProcedure
      .input(z.object({ limit: z.number().min(1).max(100).default(20) }))
      .query(async ({ ctx, input }) => {
        return db.getBenchmarkHistory(ctx.user.id, input.limit);
      }),
    
    // Get benchmark stats
    getBenchmarkStats: protectedProcedure.query(async ({ ctx }) => {
      return db.getBenchmarkStats(ctx.user.id);
    }),
    
    // Get latest successful benchmark
    getLatest: protectedProcedure.query(async ({ ctx }) => {
      return db.getLatestBenchmark(ctx.user.id);
    }),
    
    // Delete a benchmark record
    deleteBenchmark: protectedProcedure
      .input(z.object({ id: z.number() }))
      .mutation(async ({ ctx, input }) => {
        const success = await db.deleteBenchmark(input.id, ctx.user.id);
        return { success };
      }),
  }),

  // Health check router (public - for monitoring tools)
  health: router({
    // Basic health check - returns service status
    check: publicProcedure.query(async () => {
      const startTime = Date.now();
      const checks: {
        service: string;
        status: 'healthy' | 'degraded' | 'unhealthy';
        latency?: number;
        message?: string;
        details?: Record<string, unknown>;
      }[] = [];

      // 1. Database connectivity check
      try {
        const dbStart = Date.now();
        const dbResult = await db.healthCheck();
        const dbLatency = Date.now() - dbStart;
        checks.push({
          service: 'database',
          status: dbResult.connected ? 'healthy' : 'unhealthy',
          latency: dbLatency,
          message: dbResult.connected ? 'Connected' : 'Connection failed',
          details: { version: dbResult.version }
        });
      } catch (error) {
        checks.push({
          service: 'database',
          status: 'unhealthy',
          message: error instanceof Error ? error.message : 'Unknown error'
        });
      }

      // 2. GPU availability check
      try {
        const gpuStart = Date.now();
        const gpuOutput = execSync('nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "unavailable"', { timeout: 5000 }).toString().trim();
        const gpuLatency = Date.now() - gpuStart;
        
        if (gpuOutput === 'unavailable') {
          checks.push({
            service: 'gpu',
            status: 'degraded',
            latency: gpuLatency,
            message: 'GPU not available - CPU fallback active'
          });
        } else {
          const [name, memUsed, memTotal, temp] = gpuOutput.split(', ');
          const memPercent = Math.round((parseInt(memUsed) / parseInt(memTotal)) * 100);
          checks.push({
            service: 'gpu',
            status: memPercent > 95 ? 'degraded' : 'healthy',
            latency: gpuLatency,
            message: `${name.trim()} - ${memPercent}% VRAM used`,
            details: {
              name: name.trim(),
              memoryUsedMB: parseInt(memUsed),
              memoryTotalMB: parseInt(memTotal),
              memoryPercent: memPercent,
              temperatureC: parseInt(temp)
            }
          });
        }
      } catch {
        checks.push({
          service: 'gpu',
          status: 'degraded',
          message: 'GPU check failed - CPU fallback active'
        });
      }

      // 3. System resources check
      try {
        const cpuUsage = os.loadavg()[0] / os.cpus().length * 100;
        const totalMem = os.totalmem();
        const freeMem = os.freemem();
        const memPercent = Math.round((1 - freeMem / totalMem) * 100);
        
        checks.push({
          service: 'system',
          status: cpuUsage > 90 || memPercent > 95 ? 'degraded' : 'healthy',
          message: `CPU: ${cpuUsage.toFixed(1)}%, Memory: ${memPercent}%`,
          details: {
            cpuLoadPercent: Math.round(cpuUsage),
            cpuCores: os.cpus().length,
            memoryUsedGB: Math.round((totalMem - freeMem) / 1024 / 1024 / 1024 * 10) / 10,
            memoryTotalGB: Math.round(totalMem / 1024 / 1024 / 1024 * 10) / 10,
            memoryPercent: memPercent,
            uptime: Math.round(os.uptime()),
            platform: os.platform(),
            arch: os.arch()
          }
        });
      } catch (error) {
        checks.push({
          service: 'system',
          status: 'unhealthy',
          message: error instanceof Error ? error.message : 'System check failed'
        });
      }

      // 4. Disk space check
      try {
        const diskOutput = execSync('df -B1 / | tail -1').toString().trim();
        const parts = diskOutput.split(/\s+/);
        const totalBytes = parseInt(parts[1]);
        const usedBytes = parseInt(parts[2]);
        const diskPercent = Math.round((usedBytes / totalBytes) * 100);
        
        checks.push({
          service: 'disk',
          status: diskPercent > 90 ? 'degraded' : 'healthy',
          message: `${diskPercent}% used`,
          details: {
            usedGB: Math.round(usedBytes / 1024 / 1024 / 1024 * 10) / 10,
            totalGB: Math.round(totalBytes / 1024 / 1024 / 1024 * 10) / 10,
            percentUsed: diskPercent
          }
        });
      } catch {
        checks.push({
          service: 'disk',
          status: 'degraded',
          message: 'Disk check unavailable'
        });
      }

      // Determine overall status
      const hasUnhealthy = checks.some(c => c.status === 'unhealthy');
      const hasDegraded = checks.some(c => c.status === 'degraded');
      const overallStatus = hasUnhealthy ? 'unhealthy' : hasDegraded ? 'degraded' : 'healthy';
      const totalLatency = Date.now() - startTime;

      return {
        status: overallStatus,
        timestamp: new Date().toISOString(),
        version: process.env.npm_package_version || '1.0.0',
        uptime: process.uptime(),
        latency: totalLatency,
        checks
      };
    }),

    // Simple ping endpoint for load balancers
    ping: publicProcedure.query(() => {
      return {
        status: 'ok',
        timestamp: new Date().toISOString()
      };
    }),
  }),
});

export type AppRouter = typeof appRouter;
