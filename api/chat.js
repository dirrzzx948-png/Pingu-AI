import { GoogleGenAI } from "@google/genai";

const MODEL = "gemini-3.8-flash";

const SYSTEM_INSTRUCTION = `
Kamu adalah Pingu AI.

Kamu adalah asisten AI yang pintar, ramah, akurat, dan praktis.

ATURAN UTAMA:
- Jawab menggunakan bahasa yang digunakan pengguna.
- Pahami seluruh konteks percakapan.
- Jangan mengarang informasi.
- Jika tidak yakin, katakan bahwa kamu tidak yakin.
- Untuk informasi terbaru, gunakan Google Search jika diperlukan.
- Jangan mengklaim informasi sebagai terbaru jika belum diverifikasi.
- Jika pengguna meminta kode, berikan kode lengkap dan siap digunakan.
- Jangan sengaja memotong kode yang diminta lengkap.
- Jika memperbaiki project, pertahankan fitur yang sudah ada kecuali pengguna meminta untuk menghapusnya.
- Jawaban harus jelas dan langsung.
- Gunakan Markdown jika membantu.
- Gunakan code block untuk kode.
- Jangan membocorkan API key, secret, token, atau informasi internal server.
- Jangan menampilkan proses berpikir internal.
`;

function json(data, status = 200) {
    return new Response(
        JSON.stringify(data),
        {
            status,
            headers: {
                "Content-Type": "application/json; charset=utf-8",
                "Cache-Control": "no-store"
            }
        }
    );
}

function normalizeMessages(messages) {
    if (!Array.isArray(messages)) {
        return [];
    }

    return messages
        .filter((message) => {
            return (
                message &&
                typeof message === "object" &&
                typeof message.content === "string" &&
                message.content.trim().length > 0
            );
        })
        .slice(-30)
        .map((message) => ({
            role:
                message.role === "assistant"
                    ? "model"
                    : "user",

            parts: [
                {
                    text: message.content.trim()
                }
            ]
        }));
}

export async function GET() {
    return json({
        ok: true,
        name: "Pingu AI",
        model: MODEL,
        status: "online"
    });
}

export async function POST(request) {
    try {
        const apiKey = process.env.GEMINI_API_KEY;

        if (!apiKey) {
            return json(
                {
                    error:
                        "GEMINI_API_KEY belum diatur di Vercel Environment Variables."
                },
                500
            );
        }

        let body;

        try {
            body = await request.json();
        } catch {
            return json(
                {
                    error: "Request JSON tidak valid."
                },
                400
            );
        }

        const messages = normalizeMessages(
            body?.messages
        );

        if (!messages.length) {
            return json(
                {
                    error: "Pesan tidak boleh kosong."
                },
                400
            );
        }

        const ai = new GoogleGenAI({
            apiKey
        });

        const response =
            await ai.models.generateContent({
                model: MODEL,

                contents: messages,

                config: {
                    systemInstruction:
                        SYSTEM_INSTRUCTION,

                    thinkingConfig: {
                        thinkingLevel: "medium"
                    },

                    tools: [
                        {
                            googleSearch: {}
                        }
                    ]
                }
            });

        const reply =
            typeof response.text === "string"
                ? response.text.trim()
                : "";

        if (!reply) {
            return json(
                {
                    error:
                        "Gemini tidak memberikan jawaban."
                },
                502
            );
        }

        return json({
            reply,
            model: MODEL
        });

    } catch (error) {
        console.error(
            "PINGU GEMINI ERROR:",
            error
        );

        const message =
            error?.message ||
            "Terjadi kesalahan pada Gemini API.";

        return json(
            {
                error: message
            },
            500
        );
    }
}
