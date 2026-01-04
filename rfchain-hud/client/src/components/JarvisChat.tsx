import { useState, useRef, useEffect, useCallback } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  MessageSquare,
  X,
  Send,
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  Minimize2,
  Maximize2,
  Loader2,
  Bot,
  User,
} from "lucide-react";
import { trpc } from "@/lib/trpc";
import { Streamdown } from "streamdown";

interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
}

interface JarvisChatProps {
  analysisContext?: Record<string, unknown>;
}

export function JarvisChat({ analysisContext }: JarvisChatProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "Good day. I am JARVIS, your RF Signal Intelligence Assistant. I can help you analyze signal data, interpret forensic results, and answer questions about your RF analysis. How may I assist you today?",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recordingTimerRef = useRef<NodeJS.Timeout | null>(null);

  const chat = trpc.chat.send.useMutation();
  const transcribe = trpc.voice.transcribe.useMutation();

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Keyboard shortcut to open chat
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "j") {
        e.preventDefault();
        setIsOpen((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Focus input when chat opens
  useEffect(() => {
    if (isOpen && !isMinimized) {
      inputRef.current?.focus();
    }
  }, [isOpen, isMinimized]);

  // Cleanup recording timer on unmount
  useEffect(() => {
    return () => {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
    };
  }, []);

  const sendMessage = useCallback(async (messageText?: string) => {
    const text = messageText || input.trim();
    if (!text || isLoading) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content: text,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    // Add placeholder for assistant response
    const assistantId = `assistant-${Date.now()}`;
    setMessages((prev) => [
      ...prev,
      {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: new Date(),
        isStreaming: true,
      },
    ]);

    try {
      const response = await chat.mutateAsync({
        message: text,
        context: analysisContext,
      });

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId
            ? { ...msg, content: response.response, isStreaming: false }
            : msg
        )
      );

      // Speak the response if not muted
      if (!isMuted && response.response) {
        speak(response.response);
      }
    } catch (error) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId
            ? {
                ...msg,
                content: "I apologize, but I encountered an error processing your request. Please try again.",
                isStreaming: false,
              }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading, chat, analysisContext, isMuted]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const speak = (text: string) => {
    if ("speechSynthesis" in window) {
      // Cancel any ongoing speech
      speechSynthesis.cancel();
      
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 0.9;
      utterance.volume = 0.8;
      
      // Try to find a British English voice
      const voices = speechSynthesis.getVoices();
      const britishVoice = voices.find(
        (v) => v.lang === "en-GB" && v.name.includes("Male")
      ) || voices.find((v) => v.lang === "en-GB");
      if (britishVoice) {
        utterance.voice = britishVoice;
      }
      
      speechSynthesis.speak(utterance);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Try to use webm format, fallback to other formats
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') 
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/webm')
          ? 'audio/webm'
          : 'audio/mp4';
      
      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = async () => {
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());
        
        // Clear recording timer
        if (recordingTimerRef.current) {
          clearInterval(recordingTimerRef.current);
          recordingTimerRef.current = null;
        }
        setRecordingTime(0);
        
        // Process the recorded audio
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
        await processAudio(audioBlob);
      };
      
      mediaRecorder.start(100); // Collect data every 100ms
      setIsListening(true);
      
      // Start recording timer
      setRecordingTime(0);
      recordingTimerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
      
    } catch (error) {
      console.error("Error starting recording:", error);
      alert("Could not access microphone. Please check your permissions.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
      setIsListening(false);
    }
  };

  const processAudio = async (audioBlob: Blob) => {
    setIsTranscribing(true);
    
    try {
      // Upload audio to server
      const formData = new FormData();
      formData.append("audio", audioBlob, "recording.webm");
      
      const uploadResponse = await fetch("/api/audio/upload", {
        method: "POST",
        body: formData,
      });
      
      if (!uploadResponse.ok) {
        throw new Error("Failed to upload audio");
      }
      
      const { url: audioUrl } = await uploadResponse.json();
      
      // Transcribe using Whisper API
      const result = await transcribe.mutateAsync({
        audioUrl,
        language: "en",
        prompt: "Transcribe the user's voice command for RF signal analysis. The user may ask about signal metrics, forensic reports, modulation types, or analysis results.",
      });
      
      if (result.text) {
        // Set the transcribed text as input and send it
        setInput(result.text);
        // Small delay to show the transcribed text before sending
        setTimeout(() => {
          sendMessage(result.text);
        }, 300);
      }
      
    } catch (error) {
      console.error("Transcription error:", error);
      setMessages((prev) => [
        ...prev,
        {
          id: `error-${Date.now()}`,
          role: "assistant",
          content: "I apologize, but I couldn't process your voice input. Please try again or type your question.",
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsTranscribing(false);
    }
  };

  const toggleListening = () => {
    if (isListening) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const formatRecordingTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full bg-primary/20 border-2 border-primary flex items-center justify-center hover:bg-primary/30 transition-all jarvis-glow-pulse group"
        title="Open JARVIS Assistant (Ctrl+J)"
      >
        <MessageSquare className="w-6 h-6 text-primary group-hover:scale-110 transition-transform" />
      </button>
    );
  }

  return (
    <div
      className={cn(
        "fixed bottom-6 right-6 z-50 jarvis-panel rounded-lg shadow-2xl transition-all duration-300",
        isMinimized ? "w-72 h-14" : "w-96 h-[500px]"
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <div className={cn(
            "w-2 h-2 rounded-full",
            isListening ? "bg-red-500 animate-pulse" : "bg-primary animate-pulse"
          )} />
          <span className="font-semibold text-sm uppercase tracking-wider jarvis-text">
            {isListening ? `RECORDING ${formatRecordingTime(recordingTime)}` : isTranscribing ? "TRANSCRIBING..." : "JARVIS"}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setIsMuted(!isMuted)}
            title={isMuted ? "Unmute" : "Mute"}
          >
            {isMuted ? (
              <VolumeX className="w-4 h-4" />
            ) : (
              <Volume2 className="w-4 h-4" />
            )}
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setIsMinimized(!isMinimized)}
          >
            {isMinimized ? (
              <Maximize2 className="w-4 h-4" />
            ) : (
              <Minimize2 className="w-4 h-4" />
            )}
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setIsOpen(false)}
          >
            <X className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {!isMinimized && (
        <>
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 h-[380px]">
            {messages.map((message) => (
              <div
                key={message.id}
                className={cn(
                  "flex gap-3",
                  message.role === "user" ? "flex-row-reverse" : ""
                )}
              >
                <div
                  className={cn(
                    "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0",
                    message.role === "user"
                      ? "bg-secondary"
                      : "bg-primary/20 border border-primary/50"
                  )}
                >
                  {message.role === "user" ? (
                    <User className="w-4 h-4" />
                  ) : (
                    <Bot className="w-4 h-4 text-primary" />
                  )}
                </div>
                <div
                  className={cn(
                    "max-w-[80%] rounded-lg p-3 text-sm",
                    message.role === "user"
                      ? "bg-secondary"
                      : "bg-primary/10 border border-primary/30"
                  )}
                >
                  {message.isStreaming ? (
                    <div className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin text-primary" />
                      <span className="text-muted-foreground">Processing...</span>
                    </div>
                  ) : (
                    <Streamdown>{message.content}</Streamdown>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-3 border-t border-border/50">
            <div className="flex gap-2">
              <Button
                variant="ghost"
                size="icon"
                className={cn(
                  "h-9 w-9 flex-shrink-0 transition-all",
                  isListening && "bg-red-500/20 text-red-500 animate-pulse",
                  isTranscribing && "bg-yellow-500/20 text-yellow-500"
                )}
                onClick={toggleListening}
                disabled={isTranscribing || isLoading}
                title={isListening ? "Stop recording" : "Start voice input"}
              >
                {isTranscribing ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : isListening ? (
                  <MicOff className="w-4 h-4" />
                ) : (
                  <Mic className="w-4 h-4" />
                )}
              </Button>
              <Input
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={isListening ? "Listening..." : isTranscribing ? "Transcribing..." : "Ask JARVIS..."}
                className="flex-1 h-9 bg-secondary/50"
                disabled={isLoading || isListening || isTranscribing}
              />
              <Button
                size="icon"
                className="h-9 w-9 flex-shrink-0"
                onClick={() => sendMessage()}
                disabled={!input.trim() || isLoading || isListening || isTranscribing}
              >
                {isLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </Button>
            </div>
            {/* Voice input hint */}
            <p className="text-xs text-muted-foreground mt-2 text-center">
              Click the mic icon or press and hold to speak
            </p>
          </div>
        </>
      )}
    </div>
  );
}
