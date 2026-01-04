CREATE TABLE `analysis_results` (
	`id` int AUTO_INCREMENT NOT NULL,
	`signalId` int NOT NULL,
	`userId` int NOT NULL,
	`sampleCount` bigint,
	`durationMs` float,
	`avgPowerDbm` float,
	`peakPowerDbm` float,
	`paprDb` float,
	`iqImbalanceDb` float,
	`snrEstimateDb` float,
	`bandwidthHz` float,
	`freqOffsetHz` float,
	`dcOffsetReal` float,
	`dcOffsetImag` float,
	`anomalies` json,
	`fullMetrics` json,
	`timeDomainUrl` varchar(1024),
	`frequencyDomainUrl` varchar(1024),
	`spectrogramUrl` varchar(1024),
	`waterfallUrl` varchar(1024),
	`constellationUrl` varchar(1024),
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `analysis_results_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `chat_messages` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`sessionId` varchar(64) NOT NULL,
	`role` enum('user','assistant','system') NOT NULL,
	`content` text NOT NULL,
	`analysisContext` json,
	`wasVoiceInput` int DEFAULT 0,
	`audioUrl` varchar(1024),
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `chat_messages_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `forensic_reports` (
	`id` int AUTO_INCREMENT NOT NULL,
	`analysisId` int NOT NULL,
	`userId` int NOT NULL,
	`rawInputHash` varchar(128),
	`postMetricsHash` varchar(128),
	`postAnomalyHash` varchar(128),
	`postDigitalHash` varchar(128),
	`postV3Hash` varchar(128),
	`preOutputHash` varchar(128),
	`rawInputHashSha3` varchar(128),
	`postMetricsHashSha3` varchar(128),
	`postAnomalyHashSha3` varchar(128),
	`postDigitalHashSha3` varchar(128),
	`postV3HashSha3` varchar(128),
	`preOutputHashSha3` varchar(128),
	`forensicPipeline` json,
	`chainOfCustody` json,
	`outputSha256` varchar(128),
	`outputSha3` varchar(128),
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `forensic_reports_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `llm_configs` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`provider` enum('builtin','openai','anthropic','local') NOT NULL DEFAULT 'builtin',
	`apiKey` varchar(512),
	`localEndpoint` varchar(512),
	`model` varchar(128),
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `llm_configs_id` PRIMARY KEY(`id`),
	CONSTRAINT `llm_configs_userId_unique` UNIQUE(`userId`)
);
--> statement-breakpoint
CREATE TABLE `signal_uploads` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`filename` varchar(255) NOT NULL,
	`originalName` varchar(255) NOT NULL,
	`fileSize` bigint NOT NULL,
	`mimeType` varchar(100),
	`s3Key` varchar(512) NOT NULL,
	`s3Url` varchar(1024) NOT NULL,
	`status` enum('pending','processing','completed','failed') NOT NULL DEFAULT 'pending',
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `signal_uploads_id` PRIMARY KEY(`id`)
);
