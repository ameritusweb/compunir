import React from 'react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  AlertTriangle, 
  CheckCircle, 
  AlertCircle,
  Info
} from 'lucide-react';

const NotificationToast = ({ type, message, onDismiss }) => {
  const icons = {
    success: CheckCircle,
    error: AlertTriangle,
    warning: AlertCircle,
    info: Info
  };

  const Icon = icons[type] || Info;

  return (
    <Alert 
      variant={type} 
      className="fixed bottom-4 right-4 w-96 animate-slide-up"
      role="alert"
    >
      <Icon className="h-4 w-4" />
      <AlertDescription>{message}</AlertDescription>
    </Alert>
  );
};

const NotificationSystem = ({ notifications, onDismiss }) => {
  return (
    <div className="fixed bottom-4 right-4 z-50 space-y-2">
      {notifications.map((notification) => (
        <NotificationToast
          key={notification.id}
          type={notification.type}
          message={notification.message}
          onDismiss={() => onDismiss(notification.id)}
        />
      ))}
    </div>
  );
};

export default NotificationSystem;