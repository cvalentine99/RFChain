CREATE TABLE `analysis_embeddings` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`analysisId` integer NOT NULL,
	`userId` integer NOT NULL,
	`contentText` text NOT NULL,
	`embedding` text NOT NULL,
	`signalFilename` text,
	`analysisDate` integer,
	`avgPowerDbm` real,
	`bandwidthHz` real,
	`hasAnomalies` integer DEFAULT 0,
	`createdAt` integer NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `analysis_embeddings_analysisId_unique` ON `analysis_embeddings` (`analysisId`);--> statement-breakpoint
CREATE TABLE `analysis_results` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`signalId` integer NOT NULL,
	`userId` integer NOT NULL,
	`sampleCount` integer,
	`durationMs` real,
	`avgPowerDbm` real,
	`peakPowerDbm` real,
	`paprDb` real,
	`iqImbalanceDb` real,
	`snrEstimateDb` real,
	`bandwidthHz` real,
	`freqOffsetHz` real,
	`dcOffsetReal` real,
	`dcOffsetImag` real,
	`anomalies` text,
	`fullMetrics` text,
	`timeDomainPath` text,
	`frequencyDomainPath` text,
	`spectrogramPath` text,
	`waterfallPath` text,
	`constellationPath` text,
	`createdAt` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `batch_jobs` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`userId` integer NOT NULL,
	`name` text,
	`totalFiles` integer NOT NULL,
	`completedFiles` integer DEFAULT 0 NOT NULL,
	`failedFiles` integer DEFAULT 0 NOT NULL,
	`status` text DEFAULT 'pending' NOT NULL,
	`sampleRate` real DEFAULT 1000000,
	`dataFormat` text DEFAULT 'complex64',
	`digitalAnalysis` integer DEFAULT 0,
	`v3Analysis` integer DEFAULT 0,
	`startedAt` integer,
	`completedAt` integer,
	`createdAt` integer NOT NULL,
	`updatedAt` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `batch_queue_items` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`batchJobId` integer NOT NULL,
	`signalUploadId` integer,
	`filename` text NOT NULL,
	`localPath` text,
	`status` text DEFAULT 'queued' NOT NULL,
	`position` integer NOT NULL,
	`analysisResultId` integer,
	`errorMessage` text,
	`startedAt` integer,
	`completedAt` integer,
	`createdAt` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `chat_messages` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`userId` integer NOT NULL,
	`sessionId` text NOT NULL,
	`role` text NOT NULL,
	`content` text NOT NULL,
	`analysisContext` text,
	`wasVoiceInput` integer DEFAULT 0,
	`audioPath` text,
	`createdAt` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `forensic_reports` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`analysisId` integer NOT NULL,
	`userId` integer NOT NULL,
	`rawInputHash` text,
	`postMetricsHash` text,
	`postAnomalyHash` text,
	`postDigitalHash` text,
	`postV3Hash` text,
	`preOutputHash` text,
	`rawInputHashSha3` text,
	`postMetricsHashSha3` text,
	`postAnomalyHashSha3` text,
	`postDigitalHashSha3` text,
	`postV3HashSha3` text,
	`preOutputHashSha3` text,
	`forensicPipeline` text,
	`chainOfCustody` text,
	`outputSha256` text,
	`outputSha3` text,
	`createdAt` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `gpu_benchmark_history` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`userId` integer NOT NULL,
	`gpuName` text,
	`gpuMemoryMb` integer,
	`cudaVersion` text,
	`driverVersion` text,
	`avgSpeedup` real,
	`maxSpeedup` real,
	`minSpeedup` real,
	`benchmarkResults` text,
	`systemInfo` text,
	`success` integer DEFAULT 1 NOT NULL,
	`errorMessage` text,
	`createdAt` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `llm_configs` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`userId` integer NOT NULL,
	`provider` text DEFAULT 'ollama' NOT NULL,
	`apiKey` text,
	`localEndpoint` text DEFAULT 'http://localhost:11434',
	`model` text DEFAULT 'llama3.2',
	`createdAt` integer NOT NULL,
	`updatedAt` integer NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `llm_configs_userId_unique` ON `llm_configs` (`userId`);--> statement-breakpoint
CREATE TABLE `rag_settings` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`userId` integer NOT NULL,
	`enabled` integer DEFAULT 1 NOT NULL,
	`similarityThreshold` real DEFAULT 0.7 NOT NULL,
	`maxResults` integer DEFAULT 5 NOT NULL,
	`autoEmbed` integer DEFAULT 1 NOT NULL,
	`createdAt` integer NOT NULL,
	`updatedAt` integer NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `rag_settings_userId_unique` ON `rag_settings` (`userId`);--> statement-breakpoint
CREATE TABLE `sessions` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`userId` integer NOT NULL,
	`token` text NOT NULL,
	`expiresAt` integer NOT NULL,
	`createdAt` integer NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `sessions_token_unique` ON `sessions` (`token`);--> statement-breakpoint
CREATE TABLE `signal_uploads` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`userId` integer NOT NULL,
	`filename` text NOT NULL,
	`originalName` text NOT NULL,
	`fileSize` integer NOT NULL,
	`mimeType` text,
	`localPath` text NOT NULL,
	`status` text DEFAULT 'pending' NOT NULL,
	`createdAt` integer NOT NULL,
	`updatedAt` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `users` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`username` text NOT NULL,
	`passwordHash` text NOT NULL,
	`name` text,
	`email` text,
	`role` text DEFAULT 'user' NOT NULL,
	`createdAt` integer NOT NULL,
	`updatedAt` integer NOT NULL,
	`lastSignedIn` integer NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `users_username_unique` ON `users` (`username`);