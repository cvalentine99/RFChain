CREATE TABLE `analysis_embeddings` (
	`id` int AUTO_INCREMENT NOT NULL,
	`analysisId` int NOT NULL,
	`userId` int NOT NULL,
	`contentText` text NOT NULL,
	`embedding` json NOT NULL,
	`signalFilename` varchar(255),
	`analysisDate` timestamp,
	`avgPowerDbm` float,
	`bandwidthHz` float,
	`hasAnomalies` int DEFAULT 0,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `analysis_embeddings_id` PRIMARY KEY(`id`),
	CONSTRAINT `analysis_embeddings_analysisId_unique` UNIQUE(`analysisId`)
);
