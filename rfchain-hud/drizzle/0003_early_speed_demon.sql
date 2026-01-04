CREATE TABLE `rag_settings` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`enabled` int NOT NULL DEFAULT 1,
	`similarityThreshold` float NOT NULL DEFAULT 0.7,
	`maxResults` int NOT NULL DEFAULT 5,
	`autoEmbed` int NOT NULL DEFAULT 1,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `rag_settings_id` PRIMARY KEY(`id`),
	CONSTRAINT `rag_settings_userId_unique` UNIQUE(`userId`)
);
