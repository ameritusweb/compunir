// nodeService.js
class NodeService {
    constructor(nodeUrl = 'http://localhost:8000') {
      this.nodeUrl = nodeUrl;
    }
  
    async fetchNodeStats() {
      try {
        const response = await fetch(`${this.nodeUrl}/api/stats`);
        return await response.json();
      } catch (error) {
        console.error('Error fetching node stats:', error);
        throw error;
      }
    }
  
    async fetchGPUMetrics() {
      try {
        const response = await fetch(`${this.nodeUrl}/api/gpu/metrics`);
        return await response.json();
      } catch (error) {
        console.error('Error fetching GPU metrics:', error);
        throw error;
      }
    }
  
    async fetchJobs(status = 'all') {
      try {
        const response = await fetch(`${this.nodeUrl}/api/jobs?status=${status}`);
        return await response.json();
      } catch (error) {
        console.error('Error fetching jobs:', error);
        throw error;
      }
    }
  
    async fetchJobDetails(jobId) {
      try {
        const response = await fetch(`${this.nodeUrl}/api/jobs/${jobId}`);
        return await response.json();
      } catch (error) {
        console.error('Error fetching job details:', error);
        throw error;
      }
    }
  
    async fetchEarningsHistory(timeframe = '7d') {
      try {
        const response = await fetch(`${this.nodeUrl}/api/earnings/history?timeframe=${timeframe}`);
        return await response.json();
      } catch (error) {
        console.error('Error fetching earnings history:', error);
        throw error;
      }
    }
  
    async fetchVerificationStats() {
      try {
        const response = await fetch(`${this.nodeUrl}/api/verifications/stats`);
        return await response.json();
      } catch (error) {
        console.error('Error fetching verification stats:', error);
        throw error;
      }
    }
  }