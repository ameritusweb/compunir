import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Download, FileSpreadsheet, FileClock, FileCheck } from 'lucide-react';

const MetricsExport = ({ metrics, timeframe }) => {
  const [exporting, setExporting] = useState(false);

  const exportMetrics = async (type) => {
    setExporting(true);
    try {
      let data;
      switch (type) {
        case 'performance':
          data = preparePerformanceData(metrics.history);
          break;
        case 'earnings':
          data = prepareEarningsData(metrics.earnings);
          break;
        case 'verification':
          data = prepareVerificationData(metrics.verification);
          break;
        default:
          data = metrics.history;
      }

      // Convert data to CSV
      const csv = convertToCSV(data);
      
      // Create download
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${type}-metrics-${timeframe}-${new Date().toISOString()}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setExporting(false);
    }
  };

  const preparePerformanceData = (history) => {
    return history.map(metric => ({
      timestamp: new Date(metric.timestamp).toISOString(),
      utilization: metric.utilization,
      temperature: metric.temperature,
      memory_used: metric.memory.used,
      memory_total: metric.memory.total
    }));
  };

  const prepareEarningsData = (earnings) => {
    return earnings.history.map(earning => ({
      date: earning.date,
      amount: earning.amount,
      verified: earning.verified,
      pending: earning.pending,
      jobs_completed: earning.jobsCompleted
    }));
  };

  const prepareVerificationData = (verification) => {
    return verification.history.map(v => ({
      timestamp: new Date(v.timestamp).toISOString(),
      success_rate: v.successRate,
      verifiers: v.verifierCount,
      average_time: v.averageTime,
      consensus_rate: v.consensusRate
    }));
  };

  const convertToCSV = (data) => {
    if (!data.length) return '';
    
    const headers = Object.keys(data[0]);
    const rows = data.map(row => 
      headers.map(header => JSON.stringify(row[header])).join(',')
    );
    
    return [
      headers.join(','),
      ...rows
    ].join('\n');
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">Export Metrics</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Button
            variant="outline"
            disabled={exporting}
            onClick={() => exportMetrics('performance')}
            className="flex items-center"
          >
            <FileSpreadsheet className="h-4 w-4 mr-2" />
            Performance Data
          </Button>

          <Button
            variant="outline"
            disabled={exporting}
            onClick={() => exportMetrics('earnings')}
            className="flex items-center"
          >
            <FileClock className="h-4 w-4 mr-2" />
            Earnings History 
          </Button>

          <Button
            variant="outline"
            disabled={exporting}
            onClick={() => exportMetrics('verification')}
            className="flex items-center"
          >
            <FileCheck className="h-4 w-4 mr-2" />
            Verification Stats
          </Button>
        </div>

        <div className="mt-4 text-xs text-muted-foreground">
          Exports include data for the selected timeframe ({timeframe})
        </div>
      </CardContent>
    </Card>
  );
};

export default MetricsExport;