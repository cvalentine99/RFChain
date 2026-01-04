import type { CreateExpressContextOptions } from "@trpc/server/adapters/express";
import type { User } from "../../drizzle/schema";
import * as db from "../db";
import { ENV } from "./env";

export type TrpcContext = {
  req: CreateExpressContextOptions["req"];
  res: CreateExpressContextOptions["res"];
  user: User | null;
};

export async function createContext(
  opts: CreateExpressContextOptions
): Promise<TrpcContext> {
  let user: User | null = null;

  // Primary: Local session token authentication
  const sessionToken = opts.req.cookies?.session_token;
  if (sessionToken) {
    try {
      user = await db.validateSession(sessionToken);
      if (user) {
        return { req: opts.req, res: opts.res, user };
      }
    } catch (error) {
      // Session validation failed, continue
      user = null;
    }
  }

  // Secondary: OAuth (only if configured and local auth failed)
  // This is optional for self-hosted deployments
  if (!user && ENV.oAuthServerUrl && ENV.appId) {
    try {
      // Dynamically import SDK only if OAuth is configured
      const { sdk } = await import("./sdk");
      user = await sdk.authenticateRequest(opts.req);
    } catch (error) {
      // OAuth authentication failed or not configured
      // This is expected for self-hosted deployments
      user = null;
    }
  }

  return {
    req: opts.req,
    res: opts.res,
    user,
  };
}

// Helper function to authenticate REST requests (for /api/* endpoints)
export async function authenticateRestRequest(req: any): Promise<User | null> {
  // Try local session token first
  const sessionToken = req.cookies?.session_token;
  if (sessionToken) {
    try {
      const user = await db.validateSession(sessionToken);
      if (user) return user;
    } catch (error) {
      // Continue to OAuth fallback
    }
  }

  // Try OAuth only if configured
  if (ENV.oAuthServerUrl && ENV.appId) {
    try {
      const { sdk } = await import("./sdk");
      return await sdk.authenticateRequest(req);
    } catch (error) {
      // OAuth failed
    }
  }

  return null;
}
