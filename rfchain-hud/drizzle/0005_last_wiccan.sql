CREATE TABLE `gpu_benchmark_history` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`gpuName` varchar(255),
	`gpuMemoryMb` int,
	`cudaVersion` varchar(32),
	`driverVersion` varchar(32),
	`avgSpeedup` float,
	`maxSpeedup` float,
	`minSpeedup` float,
	`benchmarkResults` json,
	`systemInfo` json,
	`success` int NOT NULL DEFAULT 1,
	`errorMessage` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `gpu_benchmark_history_id` PRIMARY KEY(`id`)
);
