CREATE TABLE `batch_jobs` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`name` varchar(255),
	`totalFiles` int NOT NULL,
	`completedFiles` int NOT NULL DEFAULT 0,
	`failedFiles` int NOT NULL DEFAULT 0,
	`status` enum('pending','processing','completed','failed','cancelled') NOT NULL DEFAULT 'pending',
	`sampleRate` float DEFAULT 1000000,
	`dataFormat` varchar(32) DEFAULT 'complex64',
	`digitalAnalysis` int DEFAULT 0,
	`v3Analysis` int DEFAULT 0,
	`startedAt` timestamp,
	`completedAt` timestamp,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `batch_jobs_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `batch_queue_items` (
	`id` int AUTO_INCREMENT NOT NULL,
	`batchJobId` int NOT NULL,
	`signalUploadId` int,
	`filename` varchar(255) NOT NULL,
	`localPath` varchar(1024),
	`status` enum('queued','processing','completed','failed') NOT NULL DEFAULT 'queued',
	`position` int NOT NULL,
	`analysisResultId` int,
	`errorMessage` text,
	`startedAt` timestamp,
	`completedAt` timestamp,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `batch_queue_items_id` PRIMARY KEY(`id`)
);
