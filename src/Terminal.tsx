import { useState, useEffect, useRef } from 'react';
import type { FormEvent, MouseEvent } from 'react';

type HistoryEntry = {
    role: string;
    text: string;
    type: 'text' | 'image';
};

// ── Provider Registry ────────────────────────────────────────────────
type ProviderConfig = {
    id: string;
    name: string;
    model: string;
    supportsImages: boolean;
    buildTextRequest: (prompt: string, apiKey: string) => { url: string; options: RequestInit };
    parseTextResponse: (data: Record<string, unknown>) => string;
    buildImageRequest?: (prompt: string, apiKey: string) => { url: string; options: RequestInit };
    parseImageResponse?: (data: Record<string, unknown>) => string;
};

const PROVIDERS: Record<string, ProviderConfig> = {
    openai: {
        id: 'openai',
        name: 'OpenAI  (GPT-4o-mini)',
        model: 'gpt-4o-mini',
        supportsImages: true,
        buildTextRequest: (prompt, apiKey) => ({
            url: 'https://api.openai.com/v1/chat/completions',
            options: {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
                body: JSON.stringify({ model: 'gpt-4o-mini', messages: [{ role: 'user', content: prompt }] })
            }
        }),
        parseTextResponse: (data) => {
            const d = data as { choices: { message: { content: string } }[] };
            return d.choices[0].message.content;
        },
        buildImageRequest: (prompt, apiKey) => ({
            url: 'https://api.openai.com/v1/images/generations',
            options: {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
                body: JSON.stringify({ prompt, n: 1, size: '512x512' })
            }
        }),
        parseImageResponse: (data) => {
            const d = data as { data: { url: string }[] };
            return d.data[0].url;
        }
    },
    gemini: {
        id: 'gemini',
        name: 'Google  (Gemini 2.5 Flash)',
        model: 'gemini-2.5-flash',
        supportsImages: true,
        buildTextRequest: (prompt, apiKey) => ({
            url: `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${apiKey}`,
            options: {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] })
            }
        }),
        parseTextResponse: (data) => {
            const d = data as { candidates: { content: { parts: { text: string }[] } }[] };
            return d.candidates[0].content.parts[0].text;
        },
        buildImageRequest: (prompt, apiKey) => ({
            url: `https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict?key=${apiKey}`,
            options: {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    instances: [{ prompt }],
                    parameters: { sampleCount: 1, aspectRatio: "1:1" }
                })
            }
        }),
        parseImageResponse: (data) => {
            const d = data as { predictions: { bytesBase64Encoded: string }[] };
            return `data:image/jpeg;base64,${d.predictions[0].bytesBase64Encoded}`;
        }
    },
    claude: {
        id: 'claude',
        name: 'Anthropic (Claude Sonnet)',
        model: 'claude-sonnet-4-20250514',
        supportsImages: false,
        buildTextRequest: (prompt, apiKey) => ({
            url: 'https://api.anthropic.com/v1/messages',
            options: {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-api-key': apiKey,
                    'anthropic-version': '2023-06-01',
                    'anthropic-dangerous-direct-browser-access': 'true'
                },
                body: JSON.stringify({
                    model: 'claude-sonnet-4-20250514',
                    max_tokens: 1024,
                    messages: [{ role: 'user', content: prompt }]
                })
            }
        }),
        parseTextResponse: (data) => {
            const d = data as { content: { text: string }[] };
            return d.content[0].text;
        }
    },
    groq: {
        id: 'groq',
        name: 'Groq    (LLaMA 3.1 8B)',
        model: 'llama-3.1-8b-instant',
        supportsImages: false,
        buildTextRequest: (prompt, apiKey) => ({
            url: 'https://api.groq.com/openai/v1/chat/completions',
            options: {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
                body: JSON.stringify({ model: 'llama-3.1-8b-instant', messages: [{ role: 'user', content: prompt }] })
            }
        }),
        parseTextResponse: (data) => {
            const d = data as { choices: { message: { content: string } }[] };
            return d.choices[0].message.content;
        }
    },
    mistral: {
        id: 'mistral',
        name: 'Mistral (Mistral Small)',
        model: 'mistral-small-latest',
        supportsImages: false,
        buildTextRequest: (prompt, apiKey) => ({
            url: 'https://api.mistral.ai/v1/chat/completions',
            options: {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
                body: JSON.stringify({ model: 'mistral-small-latest', messages: [{ role: 'user', content: prompt }] })
            }
        }),
        parseTextResponse: (data) => {
            const d = data as { choices: { message: { content: string } }[] };
            return d.choices[0].message.content;
        }
    },
    cohere: {
        id: 'cohere',
        name: 'Cohere  (Command R+)',
        model: 'command-r-plus',
        supportsImages: false,
        buildTextRequest: (prompt, apiKey) => ({
            url: 'https://api.cohere.com/v2/chat',
            options: {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
                body: JSON.stringify({ model: 'command-r-plus', messages: [{ role: 'user', content: prompt }] })
            }
        }),
        parseTextResponse: (data) => {
            const d = data as { message: { content: { text: string }[] } };
            return d.message.content[0].text;
        }
    },
    nvidia: {
        id: 'nvidia',
        name: 'Groq (Llama 3 70B)',
        model: 'llama3-70b-8192',
        supportsImages: false,
        buildTextRequest: (prompt, apiKey) => ({
            url: 'https://api.groq.com/openai/v1/chat/completions',
            options: {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
                body: JSON.stringify({
                    model: 'llama3-70b-8192',
                    messages: [{ role: 'user', content: prompt }]
                })
            }
        }),
        parseTextResponse: (data) => {
            const d = data as { choices: { message: { content: string } }[] };
            return d.choices[0].message.content;
        }
    }
};

const PROVIDER_IDS = Object.keys(PROVIDERS);

// ── Initial History ──────────────────────────────────────────────────
const INITIAL_HISTORY: HistoryEntry[] = [
    { role: 'system', text: '[INIT] Boot sequence engaged. Loading kernel modules...', type: 'text' },
    { role: 'system', text: '[INIT] Mounting virtual file systems: /dev, /proc, /sys', type: 'text' },
    { role: 'system', text: '[OK] Network interface eth0 initialized. Assigned IP: 192.168.1.104', type: 'text' },
    { role: 'system', text: '[INFO] Establishing secure connection to mainframe...', type: 'text' },
    { role: 'system', text: '[SUCCESS] Bypass successful. Shell access granted.', type: 'text' },
    { role: 'system', text: 'System ready. Type "help" to see available commands.', type: 'text' }
];

export default function Terminal() {
    const [activeTab, setActiveTab] = useState<'terminal' | 'network' | 'logs' | 'sys_config'>('terminal');
    const [isGlitching, setIsGlitching] = useState(false);
    const [isRedacted, setIsRedacted] = useState(false);
    const [showRedaction, setShowRedaction] = useState(false);
    const [isRedacting, setIsRedacting] = useState(false);
    const [history, setHistory] = useState<HistoryEntry[]>(INITIAL_HISTORY);
    const [inputValue, setInputValue] = useState<string>('');
    const [apiKeys, setApiKeys] = useState<Record<string, string>>({});
    const [activeProvider, setActiveProvider] = useState<string>('openai');

    const bottomRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const isRunningRef = useRef<boolean>(false);
    const isPrankingRef = useRef<boolean>(false);
    const [popups, setPopups] = useState<{ id: number, x: number, y: number }[]>([]);

    // Auto-scroll to bottom whenever history changes
    useEffect(() => {
        if (activeTab === 'terminal') {
            bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
        }
    }, [history, activeTab]);

    // Keep focus on input for a real terminal feel
    useEffect(() => {
        if (activeTab === 'terminal') {
            inputRef.current?.focus();
        }
    }, [activeTab]);

    // Ctrl + C listener to kill processes
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.ctrlKey && e.key.toLowerCase() === 'c' && (isRunningRef.current || isPrankingRef.current)) {
                isRunningRef.current = false;
                isPrankingRef.current = false;
                setPopups([]);
                // Prefixing with [ERROR] triggers the existing red styling
                setHistory((prev) => [...prev, { role: 'system', text: '[ERROR] ^C [PROCESS TERMINATED BY USER]', type: 'text' }]);
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, []);

    const handlePrankClick = async (e: MouseEvent) => {
        e.preventDefault();
        if (isPrankingRef.current) return;
        isPrankingRef.current = true;
        let count = 0;
        while (isPrankingRef.current) {
            const x = Math.floor(Math.random() * 80);
            const y = Math.floor(Math.random() * 80);
            setPopups((prev) => [...prev.slice(-30), { id: count++, x, y }]);
            await new Promise((r) => setTimeout(r, 100));
        }
    };

    const handleDashboardClick = (e: MouseEvent) => {
        e.preventDefault();
        setActiveTab('terminal');

        const drainSequence = [
            { text: "[SYS] Connecting to Dashboard analytics...", wait: 0 },
            { text: "[WARN] Unencrypted wallet.dat detected in local storage.", wait: 600 },
            { text: "[CRITICAL] Bypassing 2FA... Extracted private keys.", wait: 1200 },
            { text: "[CRITICAL] Initiating unauthorized transfer of 4.2069 BTC to unknown address...", wait: 1800 },
            { text: "[SUCCESS] Transaction broadcasted. Wallet balance: 0.00000. BROKEEE AHHH PEOPLE.", wait: 2600 }
        ];

        drainSequence.forEach((item) => {
            setTimeout(() => {
                setHistory(prev => [...prev.slice(-300), { role: 'system', text: item.text, type: 'text' }]);
            }, item.wait);
        });

        setIsGlitching(true);
        setTimeout(() => setIsGlitching(false), 1200);
    };

    const handleNodesClick = (e: MouseEvent) => {
        e.preventDefault();
        setActiveTab('terminal');

        const scaryLines = [
            "[X-LINK] ATTEMPTING OVERFLOW AT 0x7FFF5FBFF... [SUCCESS]",
            "[FS] MOUNTING /dev/sda1 AS READ-WRITE... [FORCE]",
            "[SYS] INJECTING SHELLCODE INTO KERNEL SPACE...",
            "[NET] BYPASSING FIREWALL... PID: 4092 [ACTIVE]",
            "[AUTH] BRUTE-FORCING ROOT PASSWORD... [CRACKED]",
            "[IO] DUMPING SYSTEM LOGS TO REMOTE UPLINK...",
            "[MEM] CORRUPTING HEAP SECTS... [OVERFLOW]",
            "[PROC] TERMINATING SECURITY_DAEMON... [SIGKILL]",
            "[DEV] OVERWRITING GPT TABLE... [LOCKED]",
            "[WARN] INTRUSION DETECTION SYSTEM DISABLED.",
        ];

        const randomHex = () => Math.random().toString(16).substring(2, 10).toUpperCase();

        // Phase 2: Script Burst (~200 lines)
        let lineCount = 0;
        const totalLines = 180;
        const burstInterval = setInterval(() => {
            const burst: HistoryEntry[] = [];
            for (let i = 0; i < 15; i++) {
                const type = Math.random();
                if (type > 0.7) {
                    burst.push({ role: 'system', text: scaryLines[Math.floor(Math.random() * scaryLines.length)], type: 'text' });
                } else if (type > 0.4) {
                    burst.push({ role: 'system', text: `[DUMP] 0x${randomHex()} : ${randomHex()} ${randomHex()} ${randomHex()}`, type: 'text' });
                } else {
                    burst.push({ role: 'system', text: `[THREAD] PID ${Math.floor(Math.random() * 9000) + 1000} ::: STREAMING ENCRYPTED_PAYLOAD...`, type: 'text' });
                }
            }
            setHistory(prev => [...prev.slice(-300), ...burst]);
            lineCount += 15;
            if (lineCount >= totalLines) {
                clearInterval(burstInterval);
                startCountdown();
            }
        }, 80);

        // Phase 3-5: Countdown and Reveal
        const startCountdown = () => {
            setTimeout(() => setHistory(prev => [...prev, { role: 'system', text: '[ERROR] [CRITICAL] SCRIPT COMPLETE. INITIATING GLOBAL LOCKDOWN IN...', type: 'text' }]), 500);

            for (let i = 5; i > 0; i--) {
                setTimeout(() => {
                    setHistory(prev => [...prev, { role: 'system', text: `[SYSTEM] ${i}...`, type: 'text' }]);
                }, 1000 + (5 - i) * 1000);
            }

            setTimeout(() => {
                setHistory(prev => [...prev, { role: 'system', text: '[FATAL] LOCKING DOWN ALL LOCAL AND REMOTE SYSTEM FILES... ENCRYPTION KEY DELETED.', type: 'text' }]);
            }, 6500);

            setTimeout(() => {
                setHistory(prev => [...prev, { role: 'system', text: "[SUCCESS] SOURCE ACQUIRED: 127.0.0.1 (Wait, that's you). RELAX, IT'S A SIKE! :)", type: 'text' }]);
            }, 8500);
        };
    };

    const handleHistoryClick = (e: MouseEvent) => {
        e.preventDefault();
        setShowRedaction(true);
        setIsRedacting(false);
        setIsRedacted(true);
        setTimeout(() => setIsRedacted(false), 2500);
    };

    useEffect(() => {
        if (showRedaction) {
            const timer = setTimeout(() => {
                setIsRedacting(true);
            }, 1500);
            return () => clearTimeout(timer);
        }
    }, [showRedaction]);

    const handleCommand = async (e: FormEvent) => {
        e.preventDefault();
        if (!inputValue.trim()) return;

        const cmd = inputValue.trim();

        // 1. Echo user command to history
        setHistory((prev) => [...prev, { role: 'user', text: `pryam@secure-shell:~$ ${cmd}`, type: 'text' }]);
        setInputValue('');

        const args = cmd.split(' ');
        const baseCmd = args[0].toLowerCase();

        // 2. Command Routing Engine
        if (baseCmd === 'help') {
            setHistory((prev) => [
                ...prev,
                { role: 'system', text: '╔══════════════════════════════════════════════════════╗', type: 'text' },
                { role: 'system', text: '║  KINETIC CONSOLE — COMMAND REFERENCE                 ║', type: 'text' },
                { role: 'system', text: '╠══════════════════════════════════════════════════════╣', type: 'text' },
                { role: 'system', text: '║  whoami             → Display operator identity      ║', type: 'text' },
                { role: 'system', text: '║  clear              → Purge terminal history         ║', type: 'text' },
                { role: 'system', text: '║  providers          → List all AI neural backends     ║', type: 'text' },
                { role: 'system', text: '║  provider <name>    → Switch active AI backend        ║', type: 'text' },
                { role: 'system', text: '║  auth <prov> <key>  → Authorize a neural link         ║', type: 'text' },
                { role: 'system', text: '║  sudo <prompt>      → Route text query to active AI   ║', type: 'text' },
                { role: 'system', text: '║  sudo apt <prompt>  → Generate image                  ║', type: 'text' },
                { role: 'system', text: '║  sudo code <prompt> → Generate code via MiniMax M2.7  ║', type: 'text' },
                { role: 'system', text: '╚══════════════════════════════════════════════════════╝', type: 'text' },
            ]);
        } else if (baseCmd === 'whoami') {
            setHistory((prev) => [
                ...prev,
                { role: 'system', text: 'Developer Bio: Senior React & Cybersecurity UI Developer. Architect of the Kinetic Console.', type: 'text' }
            ]);
        } else if (baseCmd === 'clear') {
            setHistory([]);

            // ── providers: list all available AI backends ──
        } else if (baseCmd === 'providers') {
            const lines: HistoryEntry[] = [
                { role: 'system', text: '┌─────────────────────────────────────────────────┐', type: 'text' },
                { role: 'system', text: '│  REGISTERED NEURAL BACKENDS                     │', type: 'text' },
                { role: 'system', text: '├─────────────────────────────────────────────────┤', type: 'text' },
            ];
            for (const id of PROVIDER_IDS) {
                const p = PROVIDERS[id];
                const isActive = id === activeProvider ? ' ◀ ACTIVE' : '';
                const hasKey = apiKeys[id] ? '🔑 LINKED' : '🔒 NO KEY';
                const img = p.supportsImages ? ' [TXT+IMG]' : ' [TXT]';
                lines.push({ role: 'system', text: `│  ${p.name}  ${hasKey}${img}${isActive}`, type: 'text' });
            }
            lines.push({ role: 'system', text: '└─────────────────────────────────────────────────┘', type: 'text' });
            lines.push({ role: 'system', text: 'Usage: provider <name> to switch. auth <provider> <key> to link.', type: 'text' });
            setHistory((prev) => [...prev, ...lines]);

            // ── provider <name>: switch active backend ──
        } else if (baseCmd === 'provider') {
            const providerName = args[1]?.toLowerCase();
            if (!providerName) {
                const p = PROVIDERS[activeProvider];
                setHistory((prev) => [...prev,
                { role: 'system', text: `[INFO] Active backend: ${p.name} (${activeProvider})`, type: 'text' },
                { role: 'system', text: `Usage: provider <${PROVIDER_IDS.join('|')}>`, type: 'text' }
                ]);
            } else if (PROVIDERS[providerName]) {
                setActiveProvider(providerName);
                const p = PROVIDERS[providerName];
                const status = apiKeys[providerName] ? 'Neural link active.' : 'WARNING: No API key set. Use auth ' + providerName + ' <key>.';
                setHistory((prev) => [...prev,
                { role: 'system', text: `[SUCCESS] Switched to ${p.name}`, type: 'text' },
                { role: 'system', text: `[STATUS] ${status}`, type: 'text' }
                ]);
            } else {
                setHistory((prev) => [...prev,
                { role: 'system', text: `[ERROR] Unknown provider: "${providerName}"`, type: 'text' },
                { role: 'system', text: `Available: ${PROVIDER_IDS.join(', ')}`, type: 'text' }
                ]);
            }

            // ── auth: store API key for a provider ──
        } else if (baseCmd === 'auth') {
            if (args.length >= 3 && PROVIDERS[args[1].toLowerCase()]) {
                // auth <provider> <key>
                const provider = args[1].toLowerCase();
                const key = args.slice(2).join(' ');
                setApiKeys((prev) => ({ ...prev, [provider]: key }));
                setActiveProvider(provider);
                setHistory((prev) => [...prev,
                { role: 'system', text: `[SUCCESS] Neural link authorized for ${PROVIDERS[provider].name}.`, type: 'text' },
                { role: 'system', text: `[INFO] Switched active backend to ${provider}.`, type: 'text' }
                ]);
            } else if (args.length === 2) {
                // Backward compatible: auth <key> → defaults to active provider
                const key = args[1];
                setApiKeys((prev) => ({ ...prev, [activeProvider]: key }));
                setHistory((prev) => [...prev,
                { role: 'system', text: `[SUCCESS] Neural link authorized for ${PROVIDERS[activeProvider].name}. Key stored in volatile memory.`, type: 'text' }
                ]);
            } else {
                setHistory((prev) => [...prev,
                { role: 'system', text: '[ERROR] Usage: auth <provider> <key>  OR  auth <key>', type: 'text' },
                { role: 'system', text: `Providers: ${PROVIDER_IDS.join(', ')}`, type: 'text' }
                ]);
            }

            // ── sudo: AI query routing ──
        } else if (baseCmd === 'sudo') {
            const isApt = args[1]?.toLowerCase() === 'apt';
            const isCode = args[1]?.toLowerCase() === 'code';
            const isScriptKiddie = args[1]?.toLowerCase() === 'script' && args[2]?.toLowerCase() === 'kiddie';

            if (isScriptKiddie) {
                isRunningRef.current = true;
                const hackerStrings = [
                    '[SYS] Bypassing NSA firewall routing...',
                    '[0x8F3A2] Decrypting RSA-4096 payload...',
                    '[WARN] Kernel panic overridden, Injecting SQL payload...',
                    '[TRACE] Tracing IP address...',
                    '[HACK] Cracking WPA3 handshake...',
                    '[SYS] Disabling mainframe security protocols...',
                    '[NET] Establishing reverse shell...',
                    '[0x11B4] Overwriting boot sector...',
                    '[WARN] Intrusion detection bypassed...',
                    '[TRACE] Routing through 7 proxies...',
                    '[HACK] Uploading rootkit...',
                    '[SYS] Dumping SAM databases...',
                    '[NET] Escalating privileges to SYSTEM...',
                    '[0x99F0] Compiling exploit payload...',
                    '[WARN] Firewall breach detected, masking MAC...',
                    '[TRACE] Locating offshore data centers...',
                    '[HACK] Brute-forcing admin credentials...',
                    '[SYS] Accessing restricted quantum node...',
                    '[NET] Initializing DDoS botnet swarm...'
                ];

                (async () => {
                    while (isRunningRef.current) {
                        const baseStr = hackerStrings[Math.floor(Math.random() * hackerStrings.length)];
                        const hex = ' 0x' + Math.floor(Math.random() * 16777215).toString(16).toUpperCase().padStart(6, '0');
                        setHistory((prev) => [...prev.slice(-100), { role: 'system', text: baseStr + hex, type: 'text' }]);
                        await new Promise((r) => setTimeout(r, 60));
                    }
                })();
                return;
            }

            const provider = PROVIDERS[activeProvider];
            const currentKey = apiKeys[activeProvider];

            if (!currentKey) {
                setHistory((prev) => [...prev,
                { role: 'system', text: `[ERROR] No API key for ${provider.name}. Use 'auth ${activeProvider} <key>' first.`, type: 'text' }
                ]);
                return;
            }

            } else if (isCode) {
                // --- CODE GENERATION LOGIC (GROQ LPU) ---
                const prompt = args.slice(2).join(' ');
                if (!prompt) {
                    setHistory((prev) => [...prev, { role: 'system', text: '[ERROR] Missing prompt. Usage: sudo code <prompt>', type: 'text' }]);
                    return;
                }

                setHistory((prev) => [...prev, { role: 'system', text: '[GROQ LPU MATRIX] Compiling request at hyper-speed...', type: 'text' }]);

                try {
                    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${apiKeys['nvidia'] || currentKey}`
                        },
                        body: JSON.stringify({
                            model: "llama3-70b-8192",
                            messages: [
                                { role: "system", content: "You are an elite senior software engineer. Output only clean, highly optimized code with brief, hacker-style technical comments. Do not use markdown backticks in the final output, just return raw formatted text so it looks natural in a terminal." },
                                { role: "user", content: prompt }
                            ],
                            temperature: 1,
                            top_p: 0.95,
                            max_tokens: 8192
                        })
                    });

                    if (!response.ok) {
                        const errBody = await response.text().catch(() => '');
                        if (response.status === 502 || errBody.includes('<html')) {
                            throw new Error('502 GATEWAY OFFLINE: GROQ upstream node unreachable. The model "llama3-70b-8192" may be offline or invalid.');
                        }
                        throw new Error(`${response.status} ${response.statusText}${errBody ? ': ' + errBody.slice(0, 200) : ''}`);
                    }
                    const data = await response.json();
                    const textReply = data.choices[0].message.content;

                    setHistory((prev) => [...prev, { role: 'system', text: `[CODE_RESPONSE]:\n${textReply}`, type: 'text' }]);
                } catch (error) {
                    const e = error as Error;
                    setHistory((prev) => [...prev, { role: 'system', text: `[ERROR] ${e.message}`, type: 'text' }]);
                }
            } else {
                // --- TEXT GENERATION LOGIC ---
                const prompt = args.slice(1).join(' ');
                if (!prompt) {
                    setHistory((prev) => [...prev, { role: 'system', text: '[ERROR] Missing prompt. Usage: sudo <prompt>', type: 'text' }]);
                    return;
                }

                setHistory((prev) => [...prev, { role: 'system', text: `[GROQ LPU MATRIX] Routing query at hyper-speed...`, type: 'text' }]);

                try {
                    const { url, options } = provider.buildTextRequest(prompt, currentKey);
                    const response = await fetch(url, options);

                    if (!response.ok) {
                        const errBody = await response.text().catch(() => '');
                        if (response.status === 502 || errBody.includes('<html')) {
                            throw new Error('502 GATEWAY OFFLINE: GROQ upstream node unreachable. The model "llama3-70b-8192" may be offline or invalid.');
                        }
                        throw new Error(`${response.status} ${response.statusText}${errBody ? ': ' + errBody.slice(0, 200) : ''}`);
                    }
                    const data = await response.json();
                    const textReply = provider.parseTextResponse(data);

                    setHistory((prev) => [...prev, { role: 'system', text: `[AI_RESPONSE]: ${textReply}`, type: 'text' }]);
                } catch (error) {
                    const e = error as Error;
                    setHistory((prev) => [...prev, { role: 'system', text: `[ERROR] ${e.message}`, type: 'text' }]);
                }
            }
        } else {
            setHistory((prev) => [...prev, { role: 'system', text: `bash: ${baseCmd}: command not found`, type: 'text' }]);
        }
    };

    const projectNodes = [
        { id: "NODE_01", name: "F1_TELEMETRY_ENGINE", status: "DATA_STREAM_ACTIVE", protocols: ["Python", "Pandas", "WebSockets"], link: "#", desc: "Real-time kinetic data simulation and analysis." },
        { id: "NODE_02", name: "KINETIC_CONSOLE", status: "LOCALHOST_SECURE", protocols: ["React", "Tailwind", "MiniMax AI"], link: "#", desc: "Zero-retention neural link AI wrapper. (Current Instance)" },
        { id: "NODE_03", name: "PROJECT_CLASSIFIED", status: "PENDING_DEPLOYMENT", protocols: ["ENCRYPTED", "ENCRYPTED"], link: "#", desc: "Awaiting encrypted payload upload. Access restricted." }
    ];

    const experienceLogs = [
        { time: "CURRENT", process: "SYS_ARCHITECT", level: "ACTIVE", message: "Architecting and building scalable Enterprise Resource Planning (ERP) and HaaS (Hardware-as-a-Service) platforms." },
        { time: "ONGOING", process: "ENTERPRISE_DEV", level: "INFO", message: "Hands-on experience deploying, maintaining, and scaling enterprise-level software infrastructure." },
        { time: "ONGOING", process: "GIT_DAEMON", level: "COMMIT", message: "Active open-source contributor. Continuously pushing optimized code to global GitHub repositories." },
        { time: "2022.XX", process: "UPLINK_ESTABLISHED", level: "AUTH", message: "Accepted into the Google Developer Program. Expanded neural and technical capacity." }
    ];

    return (
        <div className={`bg-background text-on-surface antialiased h-screen h-[100dvh] w-full overflow-hidden flex flex-col selection:bg-primary selection:text-on-primary transition-all duration-75 ${isGlitching ? 'animate-glitch' : ''}`}>
            {/* TopAppBar */}
            <header className="flex justify-between items-center w-full px-6 py-4 bg-zinc-950 docked full-width top-0 border-none glow-sm shadow-[#6bfb9a]/10 z-50 fixed">
                <div className="flex items-center gap-8">
                    <div className="text-xl font-black text-[#6bfb9a] tracking-widest font-headline">KINETIC_CONSOLE_V4.2</div>
                    <nav className="hidden md:flex gap-6">
                        <a onClick={() => setActiveTab('terminal')} className={`font-['Space_Grotesk'] tracking-tighter uppercase font-bold text-xs transition-all px-2 py-1 flicker-on-click active:scale-95 cursor-pointer ${activeTab === 'terminal' ? 'text-[#6bfb9a] border-b-2 border-[#6bfb9a]' : 'text-zinc-500 hover:bg-[#6bfb9a]/10 hover:text-white'}`}>TERMINAL</a>
                        <a onClick={() => setActiveTab('network')} className={`font-['Space_Grotesk'] tracking-tighter uppercase font-bold text-xs transition-all px-2 py-1 flicker-on-click active:scale-95 cursor-pointer ${activeTab === 'network' ? 'text-[#6bfb9a] border-b-2 border-[#6bfb9a]' : 'text-zinc-500 hover:bg-[#6bfb9a]/10 hover:text-white'}`}>NETWORK</a>
                        <a onClick={() => setActiveTab('logs')} className={`font-['Space_Grotesk'] tracking-tighter uppercase font-bold text-xs transition-all px-2 py-1 flicker-on-click active:scale-95 cursor-pointer ${activeTab === 'logs' ? 'text-[#6bfb9a] border-b-2 border-[#6bfb9a]' : 'text-zinc-500 hover:bg-[#6bfb9a]/10 hover:text-white'}`}>LOGS</a>
                        <a onClick={() => setActiveTab('sys_config')} className={`font-['Space_Grotesk'] tracking-tighter uppercase font-bold text-xs transition-all px-2 py-1 flicker-on-click active:scale-95 cursor-pointer ${activeTab === 'sys_config' ? 'text-[#6bfb9a] border-b-2 border-[#6bfb9a]' : 'text-zinc-500 hover:bg-[#6bfb9a]/10 hover:text-white'}`}>SYS_CONFIG</a>
                    </nav>
                </div>
                <div className="flex items-center gap-4">
                    <div className="relative hidden md:block">
                        <input className="bg-surface-container-lowest border-b-2 border-primary text-primary text-xs font-label tracking-widest focus:outline-none focus:ring-0 px-2 py-1 w-48 placeholder-outline-variant flicker-50ms" placeholder="SEARCH_NETWORK..." type="text" />
                    </div>
                    <button className="text-zinc-500 hover:text-[#6bfb9a] transition-colors"><span className="material-symbols-outlined">settings_input_component</span></button>
                    <button className="text-zinc-500 hover:text-[#6bfb9a] transition-colors"><span className="material-symbols-outlined">terminal</span></button>
                    <button className="text-zinc-500 hover:text-[#6bfb9a] transition-colors"><span className="material-symbols-outlined">sensors</span></button>
                </div>
            </header>

            {/* Terminal View */}
            {activeTab === 'terminal' && (
                <div className={`flex flex-1 pt-[72px] relative overflow-hidden w-full h-full bg-surface ${isRedacted ? 'transition-all duration-500 filter blur-sm brightness-50 grayscale' : ''}`}>
                    <div className="absolute inset-0 scanline z-50 mix-blend-overlay mt-[72px] pointer-events-none"></div>
                    <div className="w-full h-full flex flex-col p-6 md:p-8 overflow-y-auto z-10 pb-24 md:pb-8">
                        <div className="max-w-7xl mx-auto w-full">
                            <div className="mb-8 text-primary font-headline text-xs tracking-widest opacity-70">
                                [SYSTEM_OK] // UPTIME: 34:12:09 // ENCRYPTION: AES-256 // NODE: LOCALHOST
                            </div>
                            <div className="mb-12 whitespace-pre font-mono text-primary text-xs md:text-sm leading-tight text-glow opacity-90 hidden sm:block">
                                {`  _  __ _____ _   _  _____  _______  _____  _____ 
 | |/ /|_   _| \\ | ||  ___||_   _||_   _|/  __ \\
 | ' /   | | |  \\| || |__    | |    | |  | /  \\/
 |  <    | | | . \` ||  __|   | |    | |  | |    
 | . \\  _| |_| |\\  || |___   | |   _| |_ | \\__/\\
 |_|\\_\\ \\___/\\_| \\_/\\____/   \\_/   \\___/  \\____/`}
                            </div>
                            <div className="space-y-4 font-mono text-sm mb-12">
                                {history.map((entry, index) => (
                                    <div key={index} className={`flex gap-4 ${entry.role === 'user' ? 'text-secondary' : 'text-on-surface'}`}>
                                        {entry.type === 'text' ? (
                                            <span className={`whitespace-pre-wrap leading-relaxed ${entry.text.includes('[ERROR]') || entry.text.includes('[CRITICAL]') ? 'text-error' :
                                                entry.text.includes('[WARN]') ? 'text-amber-500' :
                                                    entry.text.includes('[SUCCESS]') ? 'text-primary' : ''
                                                }`}>
                                                {entry.text}
                                            </span>
                                        ) : (
                                            <img src={entry.text} alt="AI Generated" className="mt-2 border-2 border-primary max-w-sm rounded shadow-glow-sm" />
                                        )}
                                    </div>
                                ))}
                                <div ref={bottomRef} />
                            </div>
                            <div className="bg-surface-container-low p-6 border-l-4 border-primary relative mb-12 shadow-glow-sm">
                                <div className="text-xs uppercase tracking-widest text-on-surface-variant mb-4">Current Directory</div>
                                <div className="text-lg text-primary font-mono text-glow">/root/sys_admin/kinetic_core/</div>
                            </div>
                            <form onSubmit={handleCommand} className="flex flex-col gap-2 relative z-20">
                                <div className="text-xs text-on-surface-variant uppercase tracking-widest mb-1">Awaiting Command input...</div>
                                <div className="flex items-center gap-3 bg-surface-container-lowest px-4 py-3 border-b-2 border-primary focus-within:bg-surface-container-low transition-colors duration-100">
                                    <span className="text-primary font-bold text-lg whitespace-nowrap">pryam@secure-shell:~$</span>
                                    <input
                                        ref={inputRef}
                                        value={inputValue}
                                        onChange={(e) => setInputValue(e.target.value)}
                                        className="bg-transparent border-none outline-none text-primary font-mono text-base w-full placeholder-outline-variant focus:ring-0 p-0"
                                        type="text"
                                        spellCheck={false}
                                    />
                                    <div className="w-2 h-5 bg-primary blinking-cursor shrink-0"></div>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
            )}

            {/* Network Scanner View */}
            {activeTab === 'network' && (
                <div className="flex flex-1 pt-[72px] relative w-full h-full">
                    <aside className="fixed left-0 top-0 flex-col pt-20 bg-zinc-900 docked h-full w-64 border-none z-40 hidden md:flex">
                        <div className="px-6 pb-8 border-b border-outline-variant/15">
                            <div className="flex items-center gap-3 mb-2">
                                <img alt="User System Operator Avatar" className="w-10 h-10 border border-outline-variant/30 grayscale opacity-80" src="https://lh3.googleusercontent.com/aida-public/AB6AXuDSUsIlBpvU7873RfRG87DMP3BGX9PMFbczpeskGqYWujdt1Tp5lqcl3ln_RPTTn4dtu5o5teOZpeKypHm1FZLiwH6qvdt3je2NTTGELZ3-_iziVSH7A_WbLGJUrfyCZNyabmpvn9yOuJv2pQoHfdxo8K6VVxwm1D7xvbDI3qtHldtaskIv81_55gYVJjIU_XuKXWm9wiWvR5uk5G3K_nJWn81JaudUCLiO_6RypPeC4WChRUN6MsV6qWwbksrs2gbIxKww3595BLk" />
                                <div>
                                    <div className="font-headline text-xs font-bold text-primary tracking-widest">OPERATOR_01</div>
                                    <div className="font-label text-[10px] text-zinc-500 tracking-widest">[STATUS: ENCRYPTED]</div>
                                </div>
                            </div>
                        </div>
                        <nav className="flex-1 flex flex-col gap-1 py-4">
                            <button type="button" onClick={handleDashboardClick} className="flex items-center gap-3 font-['Space_Grotesk'] text-[10px] uppercase tracking-[0.2em] text-zinc-500 px-4 py-3 hover:bg-zinc-800 hover:text-[#6bfb9a] glitch-hover duration-100 w-full text-left">
                                <span className="material-symbols-outlined text-base">grid_view</span>
                                DASHBOARD
                            </button>
                            <button type="button" onClick={handlePrankClick} className="flex items-center gap-3 font-['Space_Grotesk'] text-[10px] uppercase tracking-[0.2em] bg-zinc-800 text-[#6bfb9a] border-l-4 border-[#6bfb9a] px-4 py-3 glitch-hover duration-100 w-full text-left">
                                <span className="material-symbols-outlined text-base">radar</span>
                                SCANNER
                            </button>
                            <button type="button" onClick={handleNodesClick} className="flex items-center gap-3 font-['Space_Grotesk'] text-[10px] uppercase tracking-[0.2em] text-zinc-500 px-4 py-3 hover:bg-zinc-800 hover:text-[#6bfb9a] glitch-hover duration-100 w-full text-left">
                                <span className="material-symbols-outlined text-base">hub</span>
                                NODES
                            </button>
                            <button type="button" onClick={handleHistoryClick} className="flex items-center gap-3 font-['Space_Grotesk'] text-[10px] uppercase tracking-[0.2em] text-zinc-500 px-4 py-3 hover:bg-zinc-800 hover:text-[#6bfb9a] glitch-hover duration-100 w-full text-left">
                                <span className="material-symbols-outlined text-base">history</span>
                                HISTORY
                            </button>
                        </nav>
                        <div className="p-4 mt-auto border-t border-outline-variant/15">
                            <button className="w-full bg-primary text-on-primary font-label text-[10px] font-bold uppercase tracking-[0.2em] py-3 hover:bg-primary-container transition-colors relative overflow-hidden group">
                                <span className="relative z-10">INITIATE_SCAN</span>
                                <div className="absolute inset-0 bg-linear-to-b from-black/0 to-primary-container opacity-0 group-hover:opacity-100 transition-opacity"></div>
                            </button>
                        </div>
                    </aside>
                    <main className={`flex-1 md:ml-64 p-6 overflow-y-auto w-full h-full ${isRedacted ? '[&_*]:!bg-zinc-800 [&_*]:!text-zinc-800 transition-colors duration-75' : ''}`}>
                        <div className="max-w-7xl mx-auto space-y-6 pb-24 md:pb-8">
                            <div className="flex items-end justify-between border-b border-outline-variant/15 pb-4">
                                <div>
                                    <h1 className="font-headline text-display-sm text-primary font-bold tracking-tighter uppercase mb-1">NETWORK_TOPOLOGY</h1>
                                    <p className="font-label text-xs text-zinc-500 tracking-widest uppercase">[SYS_STATUS: OPTIMAL_ROUTING]</p>
                                </div>
                                <div className="font-label text-[10px] text-zinc-500 uppercase tracking-widest text-right">
                                    <div>UPTIME: 99.998%</div>
                                    <div>ACTIVE_NODES: 1,402</div>
                                </div>
                            </div>
                            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
                                {projectNodes.map((node) => (
                                    <div key={node.id} className="bg-surface-container-low border-l-4 border-primary shadow-[0_0_15px_rgba(107,251,154,0.05)] p-6 flex flex-col h-full group hover:bg-surface-container-high transition-colors">
                                        <div className="flex justify-between items-start mb-4 gap-2">
                                            <h2 className="font-headline text-primary uppercase tracking-widest font-bold text-sm">[{node.id}: {node.name}]</h2>
                                            <span className="flex items-center gap-2 font-label text-[10px] text-primary tracking-widest uppercase shrink-0">
                                                <span className="w-2 h-2 rounded-full bg-primary animate-pulse"></span>
                                                {node.status}
                                            </span>
                                        </div>
                                        <p className="text-zinc-400 font-label text-xs tracking-wider mb-6 leading-relaxed">
                                            {node.desc}
                                        </p>
                                        <div className="flex flex-wrap gap-2 mb-6">
                                            {node.protocols.map((proto, i) => (
                                                <span key={i} className="text-[10px] font-label text-primary border border-primary/30 px-2 py-1 uppercase tracking-widest bg-primary/5">
                                                    {proto}
                                                </span>
                                            ))}
                                        </div>
                                        <div className="mt-auto pt-4 border-t border-outline-variant/15">
                                            <a href={node.link} target="_blank" rel="noopener noreferrer" className="font-label text-xs text-zinc-500 hover:text-primary transition-all flex items-center gap-2 uppercase tracking-widest w-fit">
                                                &gt; ESTABLISH_CONNECTION
                                            </a>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </main>
                </div>
            )}

            {/* Logs View */}
            {activeTab === 'logs' && (
                <div className="flex flex-1 pt-[72px] relative w-full h-full">
                    <aside className="fixed left-0 top-0 h-full flex-col pt-20 w-64 border-none z-40 bg-zinc-900 hidden md:flex">
                        <div className="px-6 py-8 border-b border-outline-variant/15 mb-4">
                            <div className="flex items-center gap-4 mb-2">
                                <div className="w-10 h-10 bg-surface-container-high border border-outline-variant/30 flex items-center justify-center overflow-hidden">
                                    <img alt="User System Operator Avatar" className="w-full h-full object-cover opacity-80 mix-blend-luminosity" src="https://lh3.googleusercontent.com/aida-public/AB6AXuDzzAKIGHHrHcEkj5ocHzELwn8efJBUMtfIu1xQr4chs9r2Uy6ojkWNKpg0-XGqntJIQQYzjHdMvE6dFcGqfE7Oae7tAp2hgf9PdER7EuAWmMcAtOPyefMX4m6YV-F4qyNa4XfCO8sqrrpuY7PMjQnRdsgKNvlmtATmfwFN1XIYtp_LYsMFzfrxft2QwZEZEx6HhwZH-RsJpx1Zfiifd2n1W5Q6k2mqEO7AaROcnRbQ20LAre3hbF7vY4Vr0-5MW_m50_yfzvOMO-I" />
                                </div>
                                <div>
                                    <div className="text-[#6bfb9a] font-['Space_Grotesk'] text-sm font-bold tracking-wider">OPERATOR_01</div>
                                    <div className="text-zinc-500 font-['Space_Grotesk'] text-[10px] uppercase tracking-widest mt-1">[STATUS: ENCRYPTED]</div>
                                </div>
                            </div>
                        </div>
                        <nav className="flex-1 flex flex-col gap-1 font-['Space_Grotesk'] text-[10px] uppercase tracking-[0.2em] w-full">
                            <button type="button" onClick={handleDashboardClick} className="text-zinc-500 px-4 py-3 hover:bg-zinc-800 hover:text-[#6bfb9a] glitch-hover duration-100 flex items-center gap-3 w-full text-left">
                                <span className="material-symbols-outlined text-sm">grid_view</span>
                                DASHBOARD
                            </button>
                            <button type="button" onClick={handlePrankClick} className="text-zinc-500 px-4 py-3 hover:bg-zinc-800 hover:text-[#6bfb9a] glitch-hover duration-100 flex items-center gap-3 w-full text-left">
                                <span className="material-symbols-outlined text-sm">radar</span>
                                SCANNER
                            </button>
                            <button type="button" onClick={handleNodesClick} className="text-zinc-500 px-4 py-3 hover:bg-zinc-800 hover:text-[#6bfb9a] glitch-hover duration-100 flex items-center gap-3 w-full text-left">
                                <span className="material-symbols-outlined text-sm">hub</span>
                                NODES
                            </button>
                            <button type="button" onClick={handleHistoryClick} className="bg-zinc-800 text-[#6bfb9a] border-l-4 border-[#6bfb9a] px-4 py-3 hover:bg-zinc-800 hover:text-[#6bfb9a] glitch-hover duration-100 flex items-center gap-3 w-full text-left">
                                <span className="material-symbols-outlined text-sm">history</span>
                                HISTORY
                            </button>
                        </nav>
                        <div className="p-4 mt-auto border-t border-outline-variant/15">
                            <button className="w-full bg-[#6bfb9a] text-on-primary font-['Space_Grotesk'] text-[10px] uppercase tracking-[0.2em] font-bold py-3 hover:bg-primary-container transition-colors duration-150 flex items-center justify-center gap-2">
                                <span className="material-symbols-outlined text-sm">play_arrow</span>
                                INITIATE_SCAN
                            </button>
                        </div>
                    </aside>
                    <main className={`flex-1 md:ml-64 p-6 md:p-8 overflow-y-auto h-full w-full ${isRedacted ? '[&_*]:!bg-zinc-800 [&_*]:!text-zinc-800 transition-colors duration-75' : ''}`}>
                        <div className="max-w-[1600px] mx-auto pb-24 md:pb-8">
                            <div className="mb-12">
                                <h1 className="font-headline text-4xl md:text-5xl lg:text-6xl text-primary tracking-[-0.02em] font-bold leading-none mb-2">[OPERATOR_TIMELINE]</h1>
                                <div className="text-xs font-label text-on-surface-variant uppercase tracking-widest mb-8">Live monitoring established. Connection secure.</div>
                                <div className="bg-surface-container-lowest p-4 ghost-border flex items-center gap-4">
                                    <span className="material-symbols-outlined text-primary text-xl">search</span>
                                    <input className="w-full bg-transparent border-none text-on-surface font-label text-sm uppercase tracking-wider focus:ring-0 placeholder:text-outline-variant input-glitch border-b-2 border-primary pb-1 outline-none" placeholder="QUERY LOGS // PROCESS_ID, LEVEL, OR KEYWORD..." type="text" />
                                    <div className="items-center gap-2 ml-auto border-l border-outline-variant/30 pl-4 hidden sm:flex">
                                        <button className="text-xs font-label uppercase tracking-widest text-primary hover:bg-primary/10 px-3 py-1 transition-colors border border-primary/30">FILTERS</button>
                                        <button className="text-xs font-label uppercase tracking-widest text-on-surface hover:bg-surface-container-high px-3 py-1 transition-colors border border-outline-variant/30 flex items-center gap-1">
                                            <span className="material-symbols-outlined text-[14px]">download</span> EXPORT
                                        </button>
                                    </div>
                                </div>
                            </div>
                            <div className="w-full max-w-full overflow-x-auto">
                                <div className="w-fit min-w-full">
                                    <div className="grid grid-cols-[120px_160px_100px_minmax(300px,1fr)] gap-4 px-4 py-2 mb-4 border-b border-primary/30 text-xs font-label text-primary tracking-widest font-bold">
                                        <div>[TIMESTAMP]</div>
                                        <div>[PROCESS]</div>
                                        <div>[LEVEL]</div>
                                        <div>[MESSAGE]</div>
                                    </div>
                                    <div className="flex flex-col gap-2 font-label text-sm">
                                        {experienceLogs.map((log, i) => {
                                            let levelClasses = "text-primary border-primary";
                                            if (log.level === "ACTIVE") levelClasses = "text-[#6bfb9a] border-[#6bfb9a] shadow-[-4px_0_15px_rgba(107,251,154,0.15)]";
                                            else if (log.level === "INFO") levelClasses = "text-cyan-400 border-cyan-400 shadow-[-4px_0_15px_rgba(34,211,238,0.15)]";
                                            else if (log.level === "COMMIT") levelClasses = "text-purple-400 border-purple-400 shadow-[-4px_0_15px_rgba(192,132,252,0.15)]";
                                            else if (log.level === "AUTH") levelClasses = "text-amber-400 border-amber-400 shadow-[-4px_0_15px_rgba(251,191,36,0.15)]";

                                            return (
                                                <div key={i} className={`grid grid-cols-[120px_160px_100px_minmax(300px,1fr)] gap-4 px-4 py-4 bg-surface-container-low hover:bg-surface-container-high transition-colors items-start group border-l-2 ${levelClasses} relative overflow-hidden`}>
                                                    <div className="absolute inset-0 bg-current opacity-5 pointer-events-none"></div>
                                                    <div className="text-zinc-500 text-[11px] font-mono mt-0.5 tracking-wider relative z-10">{log.time}</div>
                                                    <div className="text-primary font-medium truncate relative z-10">{log.process}</div>
                                                    <div className="tracking-widest font-bold relative z-10">{log.level}</div>
                                                    <div className="text-zinc-300 leading-relaxed relative z-10">{log.message}</div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </main>
                </div>
            )}

            {/* System Config View */}
            {activeTab === 'sys_config' && (
                <div className="flex flex-1 pt-[72px] relative w-full h-full bg-surface">
                    <main className={`flex-1 p-6 md:p-8 overflow-y-auto h-full w-full ${isRedacted ? 'transition-all duration-500 filter blur-sm brightness-50 grayscale' : ''}`}>
                        <div className="max-w-4xl mx-auto pb-24 md:pb-8">
                            <div className="mb-8">
                                <h1 className="font-headline text-display-sm text-primary tracking-widest font-bold uppercase mb-2">[ SYS_CONFIG.JSON ]</h1>
                                <div className="text-xs font-label text-on-surface-variant uppercase tracking-widest">Operator details and uplink parameters.</div>
                            </div>
                            <div className="bg-[#0c0c0e] border border-outline-variant/30 rounded-lg shadow-2xl overflow-hidden font-mono text-sm md:text-base flex">
                                {/* Line numbers */}
                                <div className="flex flex-col text-right px-4 py-6 bg-black/40 text-zinc-600 border-r border-outline-variant/10 select-none">
                                    {Array.from({ length: 16 }).map((_, i) => (
                                        <span key={i + 1} className="leading-loose">{i + 1}</span>
                                    ))}
                                </div>
                                {/* Code Content */}
                                <div className="flex-1 p-6 overflow-x-auto">
                                    <div className="whitespace-pre leading-loose text-zinc-300">
                                        <span className="text-zinc-500">{"{"}</span>{'\n'}
                                        {'  '}<span className="text-blue-400">"operator_id"</span><span className="text-zinc-500">{": "}</span><span className="text-green-400">"Pryam"</span><span className="text-zinc-500">{","}</span>{'\n'}
                                        {'  '}<span className="text-blue-400">"clearance_level"</span><span className="text-zinc-500">{": "}</span><span className="text-green-400">"ROOT"</span><span className="text-zinc-500">{","}</span>{'\n'}
                                        {'  '}<span className="text-blue-400">"status"</span><span className="text-zinc-500">{": "}</span><span className="text-green-400">"AVAILABLE_FOR_DEPLOYMENT"</span><span className="text-zinc-500">{","}</span>{'\n'}
                                        {'  '}<span className="text-blue-400">"system_ready"</span><span className="text-zinc-500">{": "}</span><span className="text-orange-400">true</span><span className="text-zinc-500">{","}</span>{'\n'}
                                        {'  '}<span className="text-blue-400">"uptime_days"</span><span className="text-zinc-500">{": "}</span><span className="text-orange-400">1042</span><span className="text-zinc-500">{","}</span>{'\n'}
                                        {'  '}<span className="text-blue-400">"core_protocols"</span><span className="text-zinc-500">{": ["}</span>{'\n'}
                                        {'    '}<span className="text-green-400">"React"</span><span className="text-zinc-500">{", "}</span>{'\n'}
                                        {'    '}<span className="text-green-400">"TypeScript"</span><span className="text-zinc-500">{", "}</span>{'\n'}
                                        {'    '}<span className="text-green-400">"Python"</span><span className="text-zinc-500">{", "}</span>{'\n'}
                                        {'    '}<span className="text-green-400">"Node.js"</span><span className="text-zinc-500">{", "}</span>{'\n'}
                                        {'    '}<span className="text-green-400">"Tailwind"</span>{'\n'}
                                        {'  '}<span className="text-zinc-500">{"],"}</span>{'\n'}
                                        {'  '}<span className="text-blue-400">"comms_uplink"</span><span className="text-zinc-500">{": "}</span><a href="mailto:priyamarora6116@gmail.com" className="text-green-400 hover:text-green-300 transition-colors underline decoration-green-400/30 underline-offset-4">"mailto:priyamarora6116@gmail.com"</a><span className="text-zinc-500">{","}</span>{'\n'}
                                        {'  '}<span className="text-blue-400">"github_node"</span><span className="text-zinc-500">{": "}</span><a href="https://github.com/iodroid-rob" target="_blank" rel="noopener noreferrer" className="text-green-400 hover:text-green-300 transition-colors underline decoration-green-400/30 underline-offset-4">"https://github.com/iodroid-rob"</a>{'\n'}
                                        <span className="text-zinc-500">{"}"}</span><span className="text-zinc-400 animate-pulse font-bold ml-1">_</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </main>
                </div>
            )}

            {/* Classified Redaction Modal */}
            {showRedaction && (
                <div className="z-[100] fixed inset-0 flex items-center justify-center bg-black/95 backdrop-blur-sm p-4">
                    <div className="bg-zinc-950 border border-zinc-800 p-8 md:p-12 max-w-2xl w-full text-left font-mono shadow-[0_0_50px_rgba(0,0,0,1)] relative overflow-hidden">
                        <div className="absolute top-0 right-0 p-2 text-[10px] text-zinc-700 tracing-widest uppercase">Clearance: Level 5</div>
                        <div className="text-red-600 font-bold mb-6 tracking-tighter text-sm uppercase">[ TOP SECRET ] INCIDENT REPORT: PROJECT TREADSTONE</div>

                        <div className="space-y-6 text-zinc-400 text-sm md:text-base leading-relaxed">
                            <p>
                                FILE_REF: TRD-992-ALPHA<br />
                                DATE: 04.21.2026 // 0400 HOURS
                            </p>

                            <p className="border-l-2 border-zinc-800 pl-6 italic">
                                "On 0400 hours, <span className={`px-1 transition-all duration-500 ${isRedacting ? 'bg-zinc-200 text-zinc-200' : ''}`}>operative</span> successfully
                                infiltrated the <span className={`px-1 transition-all duration-500 ${isRedacting ? 'bg-zinc-200 text-zinc-200' : ''}`}>mainframe</span>.
                                The asset was located in sector 4. Casualties were minimal, but the
                                <span className={`px-1 transition-all duration-500 ${isRedacting ? 'bg-zinc-200 text-zinc-200' : ''}`}> simulation source code</span> was exposed."
                            </p>

                            <p className="text-zinc-500 text-xs">
                                STATUS: COMPROMISED<br />
                                ACTION: TERMINATE UPLINK IMMEDIATELY.
                            </p>
                        </div>

                        <button
                            onClick={() => setShowRedaction(false)}
                            className="mt-12 w-full border border-red-900/50 text-red-600 font-bold py-4 text-[10px] tracking-[0.4em] uppercase animate-pulse hover:bg-red-600 hover:text-white transition-all duration-300"
                        >
                            [ CONNECTION SEVERED - CLICK TO CLOSE ]
                        </button>
                    </div>
                </div>
            )}

            {/* Mobile Navigation */}
            <nav className="fixed bottom-0 left-0 w-full z-50 flex justify-center items-center gap-12 px-8 bg-[#09090b]/90 backdrop-blur-xl h-12 border-t-2 border-[#6bfb9a]/20 md:hidden shadow-[0_-4px_16px_rgba(107,251,154,0.05)]">
                <button type="button" onClick={() => setActiveTab('terminal')} className={`flex items-center gap-2 ${activeTab === 'terminal' ? "text-[#6bfb9a] before:content-['>'] before:animate-pulse" : "text-zinc-600 hover:bg-[#6bfb9a]/5"} font-['Space_Grotesk'] text-[0.6875rem] font-bold tracking-widest uppercase cursor-pointer px-2 py-1 transition-colors`}>
                    <span className="material-symbols-outlined text-base" style={{ fontVariationSettings: "'FILL' 0" }}>terminal</span>
                </button>
                <button type="button" onClick={() => setActiveTab('network')} className={`flex items-center gap-2 ${activeTab === 'network' ? "text-[#6bfb9a] before:content-['>'] before:animate-pulse" : "text-zinc-600 hover:bg-[#6bfb9a]/5"} font-['Space_Grotesk'] text-[0.6875rem] font-bold tracking-widest uppercase cursor-pointer px-2 py-1 transition-colors`}>
                    <span className="material-symbols-outlined text-base" style={{ fontVariationSettings: "'FILL' 0" }}>analytics</span>
                </button>
                <button type="button" onClick={() => setActiveTab('logs')} className={`flex items-center gap-2 ${activeTab === 'logs' ? "text-[#6bfb9a] before:content-['>'] before:animate-pulse" : "text-zinc-600 hover:bg-[#6bfb9a]/5"} font-['Space_Grotesk'] text-[0.6875rem] font-bold tracking-widest uppercase cursor-pointer px-2 py-1 transition-colors`}>
                    <span className="material-symbols-outlined text-base" style={{ fontVariationSettings: "'FILL' 0" }}>subject</span>
                </button>
            </nav>

            {/* Pop-up Spam Prank */}
            {popups.map((popup) => (
                <div
                    key={popup.id}
                    className="fixed bg-zinc-900 border-2 border-red-500 shadow-[4px_4px_0px_red] w-64 p-4 flex flex-col pointer-events-none"
                    style={{ top: `${popup.y}vh`, left: `${popup.x}vw`, zIndex: 100 + popup.id }}
                >
                    <div className="text-red-500 font-bold mb-2">[CRITICAL ERROR]</div>
                    <div className="text-white mb-4">SYSTEM BREACH IMMINENT...</div>
                    <div className="text-[8px] text-zinc-600 mt-auto">(it's a prank. press ctrl+c to abort)</div>
                </div>
            ))}
        </div>
    );
}