import { ENV } from "./env";

export type Role = "system" | "user" | "assistant" | "tool" | "function";

export type TextContent = {
  type: "text";
  text: string;
};

export type ImageContent = {
  type: "image_url";
  image_url: {
    url: string;
    detail?: "auto" | "low" | "high";
  };
};

export type FileContent = {
  type: "file_url";
  file_url: {
    url: string;
    mime_type?: "audio/mpeg" | "audio/wav" | "application/pdf" | "audio/mp4" | "video/mp4" ;
  };
};

export type MessageContent = string | TextContent | ImageContent | FileContent;

export type Message = {
  role: Role;
  content: MessageContent | MessageContent[];
  name?: string;
  tool_call_id?: string;
};

export type Tool = {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
};

export type ToolChoicePrimitive = "none" | "auto" | "required";
export type ToolChoiceByName = { name: string };
export type ToolChoiceExplicit = {
  type: "function";
  function: {
    name: string;
  };
};

export type ToolChoice =
  | ToolChoicePrimitive
  | ToolChoiceByName
  | ToolChoiceExplicit;

export type InvokeParams = {
  messages: Message[];
  tools?: Tool[];
  toolChoice?: ToolChoice;
  tool_choice?: ToolChoice;
  maxTokens?: number;
  max_tokens?: number;
  outputSchema?: OutputSchema;
  output_schema?: OutputSchema;
  responseFormat?: ResponseFormat;
  response_format?: ResponseFormat;
  // Local LLM options
  localEndpoint?: string;
  model?: string;
};

export type ToolCall = {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
};

export type InvokeResult = {
  id: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: Role;
      content: string | Array<TextContent | ImageContent | FileContent>;
      tool_calls?: ToolCall[];
    };
    finish_reason: string | null;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
};

export type JsonSchema = {
  name: string;
  schema: Record<string, unknown>;
  strict?: boolean;
};

export type OutputSchema = JsonSchema;

export type ResponseFormat =
  | { type: "text" }
  | { type: "json_object" }
  | { type: "json_schema"; json_schema: JsonSchema };

const ensureArray = (
  value: MessageContent | MessageContent[]
): MessageContent[] => (Array.isArray(value) ? value : [value]);

const normalizeContentPart = (
  part: MessageContent
): TextContent | ImageContent | FileContent => {
  if (typeof part === "string") {
    return { type: "text", text: part };
  }

  if (part.type === "text") {
    return part;
  }

  if (part.type === "image_url") {
    return part;
  }

  if (part.type === "file_url") {
    return part;
  }

  throw new Error("Unsupported message content part");
};

const normalizeMessage = (message: Message) => {
  const { role, name, tool_call_id } = message;

  if (role === "tool" || role === "function") {
    const content = ensureArray(message.content)
      .map(part => (typeof part === "string" ? part : JSON.stringify(part)))
      .join("\n");

    return {
      role,
      name,
      tool_call_id,
      content,
    };
  }

  const contentParts = ensureArray(message.content).map(normalizeContentPart);

  // If there's only text content, collapse to a single string for compatibility
  if (contentParts.length === 1 && contentParts[0].type === "text") {
    return {
      role,
      name,
      content: contentParts[0].text,
    };
  }

  return {
    role,
    name,
    content: contentParts,
  };
};

const normalizeToolChoice = (
  toolChoice: ToolChoice | undefined,
  tools: Tool[] | undefined
): "none" | "auto" | ToolChoiceExplicit | undefined => {
  if (!toolChoice) return undefined;

  if (toolChoice === "none" || toolChoice === "auto") {
    return toolChoice;
  }

  if (toolChoice === "required") {
    if (!tools || tools.length === 0) {
      throw new Error(
        "tool_choice 'required' was provided but no tools were configured"
      );
    }

    if (tools.length > 1) {
      throw new Error(
        "tool_choice 'required' needs a single tool or specify the tool name explicitly"
      );
    }

    return {
      type: "function",
      function: { name: tools[0].function.name },
    };
  }

  if ("name" in toolChoice) {
    return {
      type: "function",
      function: { name: toolChoice.name },
    };
  }

  return toolChoice;
};

// Determine which LLM backend to use
type LLMBackend = 'forge' | 'ollama' | 'anthropic' | 'openai' | 'none';

function detectLLMBackend(): LLMBackend {
  // Check for Ollama (local LLM)
  if (process.env.OLLAMA_HOST) {
    return 'ollama';
  }
  
  // Check for Anthropic API
  if (process.env.ANTHROPIC_API_KEY) {
    return 'anthropic';
  }
  
  // Check for OpenAI API
  if (process.env.OPENAI_API_KEY) {
    return 'openai';
  }
  
  // Check for Forge API (Manus built-in)
  if (ENV.forgeApiUrl && ENV.forgeApiKey) {
    return 'forge';
  }
  
  return 'none';
}

const resolveApiUrl = (backend: LLMBackend): string => {
  switch (backend) {
    case 'ollama':
      const ollamaHost = process.env.OLLAMA_HOST || 'http://localhost:11434';
      return `${ollamaHost.replace(/\/$/, "")}/api/chat`;
    case 'anthropic':
      return 'https://api.anthropic.com/v1/messages';
    case 'openai':
      return 'https://api.openai.com/v1/chat/completions';
    case 'forge':
    default:
      return ENV.forgeApiUrl && ENV.forgeApiUrl.trim().length > 0
        ? `${ENV.forgeApiUrl.replace(/\/$/, "")}/v1/chat/completions`
        : "https://forge.manus.im/v1/chat/completions";
  }
};

const getApiKey = (backend: LLMBackend): string => {
  switch (backend) {
    case 'anthropic':
      return process.env.ANTHROPIC_API_KEY || '';
    case 'openai':
      return process.env.OPENAI_API_KEY || '';
    case 'forge':
      return ENV.forgeApiKey || '';
    case 'ollama':
    default:
      return ''; // Ollama doesn't need an API key
  }
};

const normalizeResponseFormat = ({
  responseFormat,
  response_format,
  outputSchema,
  output_schema,
}: {
  responseFormat?: ResponseFormat;
  response_format?: ResponseFormat;
  outputSchema?: OutputSchema;
  output_schema?: OutputSchema;
}):
  | { type: "json_schema"; json_schema: JsonSchema }
  | { type: "text" }
  | { type: "json_object" }
  | undefined => {
  const explicitFormat = responseFormat || response_format;
  if (explicitFormat) {
    if (
      explicitFormat.type === "json_schema" &&
      !explicitFormat.json_schema?.schema
    ) {
      throw new Error(
        "responseFormat json_schema requires a defined schema object"
      );
    }
    return explicitFormat;
  }

  const schema = outputSchema || output_schema;
  if (!schema) return undefined;

  if (!schema.name || !schema.schema) {
    throw new Error("outputSchema requires both name and schema");
  }

  return {
    type: "json_schema",
    json_schema: {
      name: schema.name,
      schema: schema.schema,
      ...(typeof schema.strict === "boolean" ? { strict: schema.strict } : {}),
    },
  };
};

// Ollama-specific invocation
async function invokeOllama(params: InvokeParams): Promise<InvokeResult> {
  const { messages, model } = params;
  const ollamaModel = model || process.env.OLLAMA_MODEL || 'llama3';
  const ollamaHost = process.env.OLLAMA_HOST || 'http://localhost:11434';
  
  // Convert messages to Ollama format
  const ollamaMessages = messages.map(msg => ({
    role: msg.role === 'system' ? 'system' : msg.role === 'assistant' ? 'assistant' : 'user',
    content: typeof msg.content === 'string' 
      ? msg.content 
      : Array.isArray(msg.content) 
        ? msg.content.map(c => typeof c === 'string' ? c : (c as TextContent).text).join('\n')
        : String(msg.content)
  }));
  
  const response = await fetch(`${ollamaHost}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: ollamaModel,
      messages: ollamaMessages,
      stream: false
    })
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Ollama invoke failed: ${response.status} ${response.statusText} – ${errorText}`);
  }
  
  const result = await response.json();
  
  // Convert Ollama response to standard format
  return {
    id: `ollama-${Date.now()}`,
    created: Math.floor(Date.now() / 1000),
    model: ollamaModel,
    choices: [{
      index: 0,
      message: {
        role: 'assistant',
        content: result.message?.content || ''
      },
      finish_reason: 'stop'
    }],
    usage: {
      prompt_tokens: result.prompt_eval_count || 0,
      completion_tokens: result.eval_count || 0,
      total_tokens: (result.prompt_eval_count || 0) + (result.eval_count || 0)
    }
  };
}

// Anthropic-specific invocation
async function invokeAnthropic(params: InvokeParams): Promise<InvokeResult> {
  const { messages, model, maxTokens, max_tokens } = params;
  const anthropicModel = model || 'claude-3-5-sonnet-20241022';
  const apiKey = process.env.ANTHROPIC_API_KEY;
  
  if (!apiKey) {
    throw new Error('ANTHROPIC_API_KEY is not configured');
  }
  
  // Extract system message and convert others
  let systemPrompt = '';
  const anthropicMessages = messages.filter(msg => {
    if (msg.role === 'system') {
      systemPrompt = typeof msg.content === 'string' 
        ? msg.content 
        : Array.isArray(msg.content)
          ? msg.content.map(c => typeof c === 'string' ? c : (c as TextContent).text).join('\n')
          : String(msg.content);
      return false;
    }
    return true;
  }).map(msg => ({
    role: msg.role === 'assistant' ? 'assistant' : 'user',
    content: typeof msg.content === 'string' 
      ? msg.content 
      : Array.isArray(msg.content)
        ? msg.content.map(c => typeof c === 'string' ? c : (c as TextContent).text).join('\n')
        : String(msg.content)
  }));
  
  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01'
    },
    body: JSON.stringify({
      model: anthropicModel,
      max_tokens: maxTokens || max_tokens || 4096,
      system: systemPrompt || undefined,
      messages: anthropicMessages
    })
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Anthropic invoke failed: ${response.status} ${response.statusText} – ${errorText}`);
  }
  
  const result = await response.json();
  
  // Convert Anthropic response to standard format
  return {
    id: result.id,
    created: Math.floor(Date.now() / 1000),
    model: result.model,
    choices: [{
      index: 0,
      message: {
        role: 'assistant',
        content: result.content?.[0]?.text || ''
      },
      finish_reason: result.stop_reason || 'stop'
    }],
    usage: {
      prompt_tokens: result.usage?.input_tokens || 0,
      completion_tokens: result.usage?.output_tokens || 0,
      total_tokens: (result.usage?.input_tokens || 0) + (result.usage?.output_tokens || 0)
    }
  };
}

export async function invokeLLM(params: InvokeParams): Promise<InvokeResult> {
  const backend = detectLLMBackend();
  
  if (backend === 'none') {
    throw new Error(
      'No LLM backend configured. Set one of: OLLAMA_HOST (local), ANTHROPIC_API_KEY, OPENAI_API_KEY, or BUILT_IN_FORGE_API_URL + BUILT_IN_FORGE_API_KEY'
    );
  }
  
  console.log(`[LLM] Using backend: ${backend}`);
  
  // Use specialized handlers for Ollama and Anthropic
  if (backend === 'ollama') {
    return invokeOllama(params);
  }
  
  if (backend === 'anthropic') {
    return invokeAnthropic(params);
  }
  
  // Standard OpenAI-compatible API (Forge, OpenAI)
  const {
    messages,
    tools,
    toolChoice,
    tool_choice,
    outputSchema,
    output_schema,
    responseFormat,
    response_format,
  } = params;

  const apiKey = getApiKey(backend);
  if (!apiKey) {
    throw new Error(`API key not configured for ${backend}`);
  }

  const payload: Record<string, unknown> = {
    model: params.model || (backend === 'openai' ? 'gpt-4o' : 'gemini-2.5-flash'),
    messages: messages.map(normalizeMessage),
  };

  if (tools && tools.length > 0) {
    payload.tools = tools;
  }

  const normalizedToolChoice = normalizeToolChoice(
    toolChoice || tool_choice,
    tools
  );
  if (normalizedToolChoice) {
    payload.tool_choice = normalizedToolChoice;
  }

  payload.max_tokens = params.maxTokens || params.max_tokens || 32768;
  
  // Only add thinking for Forge/Gemini
  if (backend === 'forge') {
    payload.thinking = {
      "budget_tokens": 128
    };
  }

  const normalizedResponseFormat = normalizeResponseFormat({
    responseFormat,
    response_format,
    outputSchema,
    output_schema,
  });

  if (normalizedResponseFormat) {
    payload.response_format = normalizedResponseFormat;
  }

  const response = await fetch(resolveApiUrl(backend), {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `LLM invoke failed: ${response.status} ${response.statusText} – ${errorText}`
    );
  }

  return (await response.json()) as InvokeResult;
}

// Export backend detection for status checks
export function getLLMBackend(): LLMBackend {
  return detectLLMBackend();
}
