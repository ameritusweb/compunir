import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Settings, Bell, TrendingUp, RefreshCw } from 'lucide-react';

const DashboardSettings = ({ settings, onUpdate }) => {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Dashboard Settings</CardTitle>
          <Settings className="h-4 w-4 text-muted-foreground" />
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Update Frequency */}
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Update Frequency</p>
              <p className="text-sm text-muted-foreground">
                Real-time data refresh rate
              </p>
            </div>
            <div className="flex items-center gap-2">
              {['5s', '10s', '30s', '1m'].map(rate => (
                <Button
                  key={rate}
                  variant={settings.updateRate === rate ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => onUpdate('updateRate', rate)}
                >
                  {rate}
                </Button>
              ))}
            </div>
          </div>

          {/* Notifications */}
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Notifications</p>
              <p className="text-sm text-muted-foreground">
                Alert preferences
              </p>
            </div>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Switch 
                  checked={settings.notifications.highTemp}
                  onCheckedChange={(checked) => 
                    onUpdate('notifications', { ...settings.notifications, highTemp: checked })}
                />
                <span className="text-sm">High Temperature</span>
              </div>
              <div className="flex items-center gap-2">
                <Switch 
                  checked={settings.notifications.jobCompletion}
                  onCheckedChange={(checked) => 
                    onUpdate('notifications', { ...settings.notifications, jobCompletion: checked })}
                />
                <span className="text-sm">Job Completion</span>
              </div>
              <div className="flex items-center gap-2">
                <Switch 
                  checked={settings.notifications.earnings}
                  onCheckedChange={(checked) => 
                    onUpdate('notifications', { ...settings.notifications, earnings: checked })}
                />
                <span className="text-sm">Earnings Updates</span>
              </div>
            </div>
          </div>

          {/* Performance Thresholds */}
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Alert Thresholds</p>
              <p className="text-sm text-muted-foreground">
                Performance monitoring levels
              </p>
            </div>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-sm">Temperature</span>
                <Badge>{settings.thresholds.temperature}°C</Badge>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm">Memory Usage</span>
                <Badge>{settings.thresholds.memory}%</Badge>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm">Utilization</span>
                <Badge>{settings.thresholds.utilization}%</Badge>
              </div>
            </div>
          </div>

          {/* Data Retention */}
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Data Retention</p>
              <p className="text-sm text-muted-foreground">
                Historical data storage
              </p>
            </div>
            <div className="flex items-center gap-2">
              {['7d', '30d', '90d', '1y'].map(period => (
                <Button
                  key={period}
                  variant={settings.retention === period ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => onUpdate('retention', period)}
                >
                  {period}
                </Button>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default DashboardSettings;