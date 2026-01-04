// Storage helpers with local fallback for self-hosted deployments
// Uses Manus storage proxy if configured, otherwise falls back to local filesystem

import { ENV } from './_core/env';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Local storage directory (relative to project root)
const LOCAL_STORAGE_DIR = path.join(__dirname, '..', 'storage_data');

type StorageConfig = { baseUrl: string; apiKey: string } | null;

function getStorageConfig(): StorageConfig {
  const baseUrl = ENV.forgeApiUrl;
  const apiKey = ENV.forgeApiKey;

  if (!baseUrl || !apiKey) {
    // Return null to indicate local storage should be used
    return null;
  }

  return { baseUrl: baseUrl.replace(/\/+$/, ""), apiKey };
}

function buildUploadUrl(baseUrl: string, relKey: string): URL {
  const url = new URL("v1/storage/upload", ensureTrailingSlash(baseUrl));
  url.searchParams.set("path", normalizeKey(relKey));
  return url;
}

async function buildDownloadUrl(
  baseUrl: string,
  relKey: string,
  apiKey: string
): Promise<string> {
  const downloadApiUrl = new URL(
    "v1/storage/downloadUrl",
    ensureTrailingSlash(baseUrl)
  );
  downloadApiUrl.searchParams.set("path", normalizeKey(relKey));
  const response = await fetch(downloadApiUrl, {
    method: "GET",
    headers: buildAuthHeaders(apiKey),
  });
  return (await response.json()).url;
}

function ensureTrailingSlash(value: string): string {
  return value.endsWith("/") ? value : `${value}/`;
}

function normalizeKey(relKey: string): string {
  return relKey.replace(/^\/+/, "");
}

function toFormData(
  data: Buffer | Uint8Array | string,
  contentType: string,
  fileName: string
): FormData {
  const blob =
    typeof data === "string"
      ? new Blob([data], { type: contentType })
      : new Blob([data as any], { type: contentType });
  const form = new FormData();
  form.append("file", blob, fileName || "file");
  return form;
}

function buildAuthHeaders(apiKey: string): HeadersInit {
  return { Authorization: `Bearer ${apiKey}` };
}

// Local storage implementation
function ensureLocalStorageDir(subPath: string): string {
  const fullPath = path.join(LOCAL_STORAGE_DIR, path.dirname(subPath));
  fs.mkdirSync(fullPath, { recursive: true });
  return path.join(LOCAL_STORAGE_DIR, subPath);
}

async function localStoragePut(
  relKey: string,
  data: Buffer | Uint8Array | string
): Promise<{ key: string; url: string }> {
  const key = normalizeKey(relKey);
  const filePath = ensureLocalStorageDir(key);
  
  // Convert data to Buffer if needed
  const buffer = typeof data === 'string' 
    ? Buffer.from(data) 
    : Buffer.from(data);
  
  fs.writeFileSync(filePath, buffer);
  
  // Return a local URL that can be served by the static file server
  const url = `/storage/${key}`;
  
  console.log(`[Storage] Saved locally: ${filePath} -> ${url}`);
  return { key, url };
}

async function localStorageGet(relKey: string): Promise<{ key: string; url: string }> {
  const key = normalizeKey(relKey);
  const url = `/storage/${key}`;
  return { key, url };
}

export async function storagePut(
  relKey: string,
  data: Buffer | Uint8Array | string,
  contentType = "application/octet-stream"
): Promise<{ key: string; url: string }> {
  const config = getStorageConfig();
  
  // Use local storage if cloud storage is not configured
  if (!config) {
    console.log("[Storage] Using local filesystem storage");
    return localStoragePut(relKey, data);
  }
  
  // Use cloud storage (Manus S3 proxy)
  const { baseUrl, apiKey } = config;
  const key = normalizeKey(relKey);
  const uploadUrl = buildUploadUrl(baseUrl, key);
  const formData = toFormData(data, contentType, key.split("/").pop() ?? key);
  
  try {
    const response = await fetch(uploadUrl, {
      method: "POST",
      headers: buildAuthHeaders(apiKey),
      body: formData,
    });

    if (!response.ok) {
      const message = await response.text().catch(() => response.statusText);
      throw new Error(
        `Storage upload failed (${response.status} ${response.statusText}): ${message}`
      );
    }
    const url = (await response.json()).url;
    return { key, url };
  } catch (error) {
    // Fall back to local storage on error
    console.warn("[Storage] Cloud storage failed, falling back to local:", error);
    return localStoragePut(relKey, data);
  }
}

export async function storageGet(relKey: string): Promise<{ key: string; url: string }> {
  const config = getStorageConfig();
  
  // Use local storage if cloud storage is not configured
  if (!config) {
    return localStorageGet(relKey);
  }
  
  const { baseUrl, apiKey } = config;
  const key = normalizeKey(relKey);
  
  try {
    return {
      key,
      url: await buildDownloadUrl(baseUrl, key, apiKey),
    };
  } catch (error) {
    // Fall back to local storage on error
    console.warn("[Storage] Cloud storage failed, falling back to local:", error);
    return localStorageGet(relKey);
  }
}

// Export local storage directory for static file serving
export const STORAGE_DIR = LOCAL_STORAGE_DIR;
