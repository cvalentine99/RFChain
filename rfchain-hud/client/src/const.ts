export { COOKIE_NAME, ONE_YEAR_MS } from "@shared/const";

// Check if OAuth is configured
const isOAuthConfigured = () => {
  const oauthPortalUrl = import.meta.env.VITE_OAUTH_PORTAL_URL;
  const appId = import.meta.env.VITE_APP_ID;
  return !!(oauthPortalUrl && appId && oauthPortalUrl !== '' && appId !== '');
};

// Generate login URL - returns local login page if OAuth not configured
export const getLoginUrl = () => {
  // If OAuth is not configured, use local login
  if (!isOAuthConfigured()) {
    return '/login';
  }
  
  const oauthPortalUrl = import.meta.env.VITE_OAUTH_PORTAL_URL;
  const appId = import.meta.env.VITE_APP_ID;
  const redirectUri = `${window.location.origin}/api/oauth/callback`;
  const state = btoa(redirectUri);

  const url = new URL(`${oauthPortalUrl}/app-auth`);
  url.searchParams.set("appId", appId);
  url.searchParams.set("redirectUri", redirectUri);
  url.searchParams.set("state", state);
  url.searchParams.set("type", "signIn");

  return url.toString();
};

// Check if using local auth mode
export const isLocalAuthMode = () => !isOAuthConfigured();
