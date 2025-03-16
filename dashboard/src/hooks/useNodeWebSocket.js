import { useState, useEffect, useCallback } from 'react';

const useNodeWebSocket = (url) => {
  const [socket, setSocket] = useState(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(url);

      ws.onopen = () => {
        setConnected(true);
        setError(null);
      };

      ws.onclose = () => {
        setConnected(false);
        // Attempt to reconnect after 5 seconds
        setTimeout(connect, 5000);
      };

      ws.onerror = (event) => {
        setError('WebSocket connection error');
        console.error('WebSocket error:', event);
      };

      setSocket(ws);
    } catch (err) {
      setError('Failed to create WebSocket connection');
      console.error('WebSocket connection error:', err);
    }
  }, [url]);

  useEffect(() => {
    connect();

    return () => {
      if (socket) {
        socket.close();
      }
    };
  }, [connect]);

  const subscribe = useCallback((channel, callback) => {
    if (!socket) return;

    const handler = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.channel === channel) {
          callback(data.payload);
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    };

    socket.addEventListener('message', handler);

    return () => {
      socket.removeEventListener('message', handler);
    };
  }, [socket]);

  const send = useCallback((data) => {
    if (socket && connected) {
      socket.send(JSON.stringify(data));
    }
  }, [socket, connected]);

  return {
    socket,
    connected,
    error,
    subscribe,
    send
  };
};

export default useNodeWebSocket;