import { GoogleGenAI } from "@google/genai";

const MODEL = "gemini-3.7-flash";

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store"
    }
  });
}

function cleanMessages(messages) {
  if (!Array.isArray(messages)) return [];

  return messages
    .filter(
      (message) =>
        message &&
        typeof message === "object" &&
        ["user", "assistant"].includes(message.role) &&
        typeof message.content === "string" &&
        message.content.trim()
    )
    .map((message) => ({
      role: message.role,
      content: message.content.trim()
    }));
}

function buildPrompt(messages) {
  const history = messages
    .map((message) => {
      const role =
        message.role === "assistant"
          ? "Pingu"
          : "User";

      return `${role}:\n${message.content}`;
    })
    .join("\n\n");

  return `Kamu adalah Pingu, AI assistant yang ramah, jelas, dan membantu.

Aturan:
- Jawab menggunakan bahasa pengguna.
- Gunakan konteks percakapan sebelumnya.
- Jangan mengarang informasi.
- Jika diminta kode, berikan kode lengkap dan siap digunakan.
- Jika pengguna meminta perubahan kode, pertahankan fitur yang sudah ada.
- Gunakan Markdown jika diperlukan.
- Jangan membocorkan API key atau informasi rahasia.

Percakapan:

${history}

Pingu:`;
}

function getGeminiError(error) {
  const message =
    error?.message ||
    error?.error?.message ||
    String(error);

  const status =
    error?.status ||
    error?.error?.code ||
    null;

  return {
    status,
    message
  };
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
          error: "GEMINI_API_KEY belum disetel di Vercel."
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
          error: "Request body tidak valid."
        },
        400
      );
    }

    const messages = cleanMessages(body?.messages);

    if (!messages.length) {
      return json(
        {
          error: "Pesan tidak ditemukan."
        },
        400
      );
    }

    const lastMessage =
      messages[messages.length - 1];

    if (lastMessage.role !== "user") {
      return json(
        {
          error: "Pesan terakhir harus dari user."
        },
        400
      );
    }

    const ai = new GoogleGenAI({
      apiKey
    });

    const prompt = buildPrompt(messages);

    console.log(
      `[Pingu] Request model: ${MODEL}`
    );

    const result =
      await ai.models.generateContent({
        model: MODEL,
        contents: prompt
      });

    const reply =
      typeof result?.text === "string"
        ? result.text.trim()
        : "";

    if (!reply) {
      return json(
        {
          error:
            "Gemini berhasil dipanggil tetapi tidak mengembalikan teks.",
          model: MODEL
        },
        502
      );
    }

    console.log(
      `[Pingu] Gemini response OK`
    );

    return json({
      reply,
      model: MODEL
    });

  } catch (error) {
    console.error(
      "[Pingu] Gemini error:",
      error
    );

    const geminiError =
      getGeminiError(error);

    const status =
      Number(geminiError.status) || 500;

    return json(
      {
        error: geminiError.message,
        code: status,
        model: MODEL
      },
      status >= 400 && status <= 599
        ? status
        : 500
    );
  }
}
