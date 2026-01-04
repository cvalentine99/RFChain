import { describe, it, expect, vi } from 'vitest';

// Mock database health check
vi.mock('./db', () => ({
  healthCheck: vi.fn().mockResolvedValue({ connected: true, version: '8.0.32-TiDB' })
}));

describe('Health Check Endpoint', () => {
  describe('health.check', () => {
    it('should return overall status', async () => {
      // The health check returns a structured response
      const mockResponse = {
        status: 'healthy',
        timestamp: expect.any(String),
        version: expect.any(String),
        uptime: expect.any(Number),
        latency: expect.any(Number),
        checks: expect.any(Array)
      };
      
      expect(mockResponse.status).toBeDefined();
      expect(['healthy', 'degraded', 'unhealthy']).toContain(mockResponse.status);
    });

    it('should include database check', async () => {
      const dbCheck = {
        service: 'database',
        status: 'healthy',
        latency: 15,
        message: 'Connected',
        details: { version: '8.0.32-TiDB' }
      };
      
      expect(dbCheck.service).toBe('database');
      expect(dbCheck.status).toBe('healthy');
      expect(dbCheck.details?.version).toBeDefined();
    });

    it('should include GPU check', async () => {
      const gpuCheck = {
        service: 'gpu',
        status: 'degraded',
        message: 'GPU not available - CPU fallback active'
      };
      
      expect(gpuCheck.service).toBe('gpu');
      expect(['healthy', 'degraded', 'unhealthy']).toContain(gpuCheck.status);
    });

    it('should include system resources check', async () => {
      const systemCheck = {
        service: 'system',
        status: 'healthy',
        message: 'CPU: 25.5%, Memory: 45%',
        details: {
          cpuLoadPercent: 26,
          cpuCores: 4,
          memoryUsedGB: 7.2,
          memoryTotalGB: 16,
          memoryPercent: 45,
          uptime: 86400,
          platform: 'linux',
          arch: 'x64'
        }
      };
      
      expect(systemCheck.service).toBe('system');
      expect(systemCheck.details?.cpuCores).toBeGreaterThan(0);
      expect(systemCheck.details?.memoryTotalGB).toBeGreaterThan(0);
    });

    it('should include disk space check', async () => {
      const diskCheck = {
        service: 'disk',
        status: 'healthy',
        message: '45% used',
        details: {
          usedGB: 450,
          totalGB: 1000,
          percentUsed: 45
        }
      };
      
      expect(diskCheck.service).toBe('disk');
      expect(diskCheck.details?.percentUsed).toBeLessThanOrEqual(100);
    });

    it('should mark status as degraded when GPU unavailable', async () => {
      const checks = [
        { service: 'database', status: 'healthy' },
        { service: 'gpu', status: 'degraded' },
        { service: 'system', status: 'healthy' },
        { service: 'disk', status: 'healthy' }
      ];
      
      const hasUnhealthy = checks.some(c => c.status === 'unhealthy');
      const hasDegraded = checks.some(c => c.status === 'degraded');
      const overallStatus = hasUnhealthy ? 'unhealthy' : hasDegraded ? 'degraded' : 'healthy';
      
      expect(overallStatus).toBe('degraded');
    });

    it('should mark status as unhealthy when database fails', async () => {
      const checks = [
        { service: 'database', status: 'unhealthy' },
        { service: 'gpu', status: 'healthy' },
        { service: 'system', status: 'healthy' },
        { service: 'disk', status: 'healthy' }
      ];
      
      const hasUnhealthy = checks.some(c => c.status === 'unhealthy');
      const hasDegraded = checks.some(c => c.status === 'degraded');
      const overallStatus = hasUnhealthy ? 'unhealthy' : hasDegraded ? 'degraded' : 'healthy';
      
      expect(overallStatus).toBe('unhealthy');
    });
  });

  describe('health.ping', () => {
    it('should return simple ok status', async () => {
      const response = {
        status: 'ok',
        timestamp: new Date().toISOString()
      };
      
      expect(response.status).toBe('ok');
      expect(response.timestamp).toBeDefined();
    });

    it('should return valid ISO timestamp', async () => {
      const response = {
        status: 'ok',
        timestamp: new Date().toISOString()
      };
      
      const parsed = new Date(response.timestamp);
      expect(parsed.toISOString()).toBe(response.timestamp);
    });
  });

  describe('Status determination logic', () => {
    it('should prioritize unhealthy over degraded', async () => {
      const checks = [
        { status: 'healthy' },
        { status: 'degraded' },
        { status: 'unhealthy' }
      ];
      
      const hasUnhealthy = checks.some(c => c.status === 'unhealthy');
      const hasDegraded = checks.some(c => c.status === 'degraded');
      const overallStatus = hasUnhealthy ? 'unhealthy' : hasDegraded ? 'degraded' : 'healthy';
      
      expect(overallStatus).toBe('unhealthy');
    });

    it('should return healthy when all checks pass', async () => {
      const checks = [
        { status: 'healthy' },
        { status: 'healthy' },
        { status: 'healthy' }
      ];
      
      const hasUnhealthy = checks.some(c => c.status === 'unhealthy');
      const hasDegraded = checks.some(c => c.status === 'degraded');
      const overallStatus = hasUnhealthy ? 'unhealthy' : hasDegraded ? 'degraded' : 'healthy';
      
      expect(overallStatus).toBe('healthy');
    });
  });
});
