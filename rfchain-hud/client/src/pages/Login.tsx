import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { trpc } from "@/lib/trpc";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Loader2, Zap, Shield, Lock } from "lucide-react";
import { toast } from "sonner";

export default function Login() {
  const [, setLocation] = useLocation();
  const [isLogin, setIsLogin] = useState(true);
  
  // Login form state
  const [loginUsername, setLoginUsername] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  
  // Register form state
  const [registerUsername, setRegisterUsername] = useState("");
  const [registerPassword, setRegisterPassword] = useState("");
  const [registerName, setRegisterName] = useState("");
  const [registerEmail, setRegisterEmail] = useState("");
  
  // Check if setup is needed (no users exist)
  const { data: setupStatus, isLoading: checkingSetup } = trpc.auth.needsSetup.useQuery();
  
  // Check if already logged in
  const { data: currentUser, isLoading: checkingAuth } = trpc.auth.me.useQuery();
  
  // Login mutation
  const loginMutation = trpc.auth.login.useMutation({
    onSuccess: () => {
      toast.success("Login successful!");
      setLocation("/");
    },
    onError: (error) => {
      toast.error(error.message || "Login failed");
    },
  });
  
  // Register mutation
  const registerMutation = trpc.auth.register.useMutation({
    onSuccess: (data) => {
      if (data.isFirstUser) {
        toast.success("Admin account created! You are now logged in.");
      } else {
        toast.success("Account created successfully!");
      }
      setLocation("/");
    },
    onError: (error) => {
      toast.error(error.message || "Registration failed");
    },
  });
  
  // Redirect if already logged in
  useEffect(() => {
    if (currentUser && !checkingAuth) {
      setLocation("/");
    }
  }, [currentUser, checkingAuth, setLocation]);
  
  // Auto-switch to register tab if no users exist
  useEffect(() => {
    if (setupStatus?.needsSetup) {
      setIsLogin(false);
    }
  }, [setupStatus]);
  
  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    loginMutation.mutate({
      username: loginUsername,
      password: loginPassword,
    });
  };
  
  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    registerMutation.mutate({
      username: registerUsername,
      password: registerPassword,
      name: registerName || undefined,
      email: registerEmail || undefined,
    });
  };
  
  if (checkingAuth || checkingSetup) {
    return (
      <div className="min-h-screen bg-[#0a0f14] flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-cyan-500" />
      </div>
    );
  }
  
  return (
    <div className="min-h-screen bg-[#0a0f14] flex items-center justify-center p-4">
      {/* Background effects */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl" />
      </div>
      
      <div className="relative z-10 w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-2">
            <Zap className="h-10 w-10 text-cyan-400" />
            <h1 className="text-3xl font-bold text-cyan-400 tracking-wider">RFCHAIN</h1>
          </div>
          <p className="text-gray-400 text-sm tracking-widest uppercase">Signal Intelligence System</p>
        </div>
        
        <Card className="bg-[#0d1419]/90 border-cyan-900/50 backdrop-blur-sm">
          <CardHeader className="text-center pb-2">
            <CardTitle className="text-xl text-cyan-400 flex items-center justify-center gap-2">
              <Shield className="h-5 w-5" />
              {setupStatus?.needsSetup ? "Initial Setup" : "Authentication"}
            </CardTitle>
            <CardDescription className="text-gray-400">
              {setupStatus?.needsSetup 
                ? "Create the first admin account to get started"
                : "Sign in to access the system"
              }
            </CardDescription>
          </CardHeader>
          
          <CardContent>
            {setupStatus?.needsSetup ? (
              // First user registration (admin setup)
              <form onSubmit={handleRegister} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="setup-username" className="text-gray-300">Username</Label>
                  <Input
                    id="setup-username"
                    type="text"
                    placeholder="admin"
                    value={registerUsername}
                    onChange={(e) => setRegisterUsername(e.target.value)}
                    required
                    minLength={3}
                    className="bg-[#0a0f14] border-cyan-900/50 text-white placeholder:text-gray-500 focus:border-cyan-500"
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="setup-name" className="text-gray-300">Display Name (optional)</Label>
                  <Input
                    id="setup-name"
                    type="text"
                    placeholder="Administrator"
                    value={registerName}
                    onChange={(e) => setRegisterName(e.target.value)}
                    className="bg-[#0a0f14] border-cyan-900/50 text-white placeholder:text-gray-500 focus:border-cyan-500"
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="setup-email" className="text-gray-300">Email (optional)</Label>
                  <Input
                    id="setup-email"
                    type="email"
                    placeholder="admin@example.com"
                    value={registerEmail}
                    onChange={(e) => setRegisterEmail(e.target.value)}
                    className="bg-[#0a0f14] border-cyan-900/50 text-white placeholder:text-gray-500 focus:border-cyan-500"
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="setup-password" className="text-gray-300">Password</Label>
                  <Input
                    id="setup-password"
                    type="password"
                    placeholder="••••••••"
                    value={registerPassword}
                    onChange={(e) => setRegisterPassword(e.target.value)}
                    required
                    minLength={6}
                    className="bg-[#0a0f14] border-cyan-900/50 text-white placeholder:text-gray-500 focus:border-cyan-500"
                  />
                  <p className="text-xs text-gray-500">Minimum 6 characters</p>
                </div>
                
                <Button 
                  type="submit" 
                  className="w-full bg-cyan-600 hover:bg-cyan-700 text-white"
                  disabled={registerMutation.isPending}
                >
                  {registerMutation.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Creating Admin Account...
                    </>
                  ) : (
                    <>
                      <Lock className="mr-2 h-4 w-4" />
                      Create Admin Account
                    </>
                  )}
                </Button>
              </form>
            ) : (
              // Normal login/register tabs
              <Tabs value={isLogin ? "login" : "register"} onValueChange={(v) => setIsLogin(v === "login")}>
                <TabsList className="grid w-full grid-cols-2 bg-[#0a0f14]">
                  <TabsTrigger value="login" className="data-[state=active]:bg-cyan-900/50 data-[state=active]:text-cyan-400">
                    Sign In
                  </TabsTrigger>
                  <TabsTrigger value="register" className="data-[state=active]:bg-cyan-900/50 data-[state=active]:text-cyan-400">
                    Register
                  </TabsTrigger>
                </TabsList>
                
                <TabsContent value="login" className="mt-4">
                  <form onSubmit={handleLogin} className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="login-username" className="text-gray-300">Username</Label>
                      <Input
                        id="login-username"
                        type="text"
                        placeholder="Enter username"
                        value={loginUsername}
                        onChange={(e) => setLoginUsername(e.target.value)}
                        required
                        className="bg-[#0a0f14] border-cyan-900/50 text-white placeholder:text-gray-500 focus:border-cyan-500"
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="login-password" className="text-gray-300">Password</Label>
                      <Input
                        id="login-password"
                        type="password"
                        placeholder="••••••••"
                        value={loginPassword}
                        onChange={(e) => setLoginPassword(e.target.value)}
                        required
                        className="bg-[#0a0f14] border-cyan-900/50 text-white placeholder:text-gray-500 focus:border-cyan-500"
                      />
                    </div>
                    
                    <Button 
                      type="submit" 
                      className="w-full bg-cyan-600 hover:bg-cyan-700 text-white"
                      disabled={loginMutation.isPending}
                    >
                      {loginMutation.isPending ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Signing In...
                        </>
                      ) : (
                        "Sign In"
                      )}
                    </Button>
                  </form>
                </TabsContent>
                
                <TabsContent value="register" className="mt-4">
                  <form onSubmit={handleRegister} className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="reg-username" className="text-gray-300">Username</Label>
                      <Input
                        id="reg-username"
                        type="text"
                        placeholder="Choose a username"
                        value={registerUsername}
                        onChange={(e) => setRegisterUsername(e.target.value)}
                        required
                        minLength={3}
                        className="bg-[#0a0f14] border-cyan-900/50 text-white placeholder:text-gray-500 focus:border-cyan-500"
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="reg-name" className="text-gray-300">Display Name (optional)</Label>
                      <Input
                        id="reg-name"
                        type="text"
                        placeholder="Your name"
                        value={registerName}
                        onChange={(e) => setRegisterName(e.target.value)}
                        className="bg-[#0a0f14] border-cyan-900/50 text-white placeholder:text-gray-500 focus:border-cyan-500"
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="reg-password" className="text-gray-300">Password</Label>
                      <Input
                        id="reg-password"
                        type="password"
                        placeholder="••••••••"
                        value={registerPassword}
                        onChange={(e) => setRegisterPassword(e.target.value)}
                        required
                        minLength={6}
                        className="bg-[#0a0f14] border-cyan-900/50 text-white placeholder:text-gray-500 focus:border-cyan-500"
                      />
                    </div>
                    
                    <Button 
                      type="submit" 
                      className="w-full bg-cyan-600 hover:bg-cyan-700 text-white"
                      disabled={registerMutation.isPending}
                    >
                      {registerMutation.isPending ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Creating Account...
                        </>
                      ) : (
                        "Create Account"
                      )}
                    </Button>
                    
                    <p className="text-xs text-gray-500 text-center">
                      Note: Only admins can create new user accounts
                    </p>
                  </form>
                </TabsContent>
              </Tabs>
            )}
          </CardContent>
        </Card>
        
        {/* Footer */}
        <p className="text-center text-gray-500 text-xs mt-6">
          RFChain HUD • Self-Hosted Mode • Forensic Analysis Platform
        </p>
      </div>
    </div>
  );
}
