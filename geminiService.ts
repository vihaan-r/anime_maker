
import { GoogleGenAI, Type } from "@google/genai";
import { Character, ColorMode, StoryboardPage, MangaProject, Location, VisualMedium } from "../types";

const STYLE_PROTOCOLS = {
  MANGA: `
    ART STYLE: Professional Japanese Manga (Seinen).
    - PRIMARY: Minimalist clean thin linework, airy screentones, generous white space.
    - IMPACT: High-contrast blacks and dynamic speed lines for high-intensity moments.
  `,
  DC_COMIC: `
    ART STYLE: Modern American Superhero Comic (DC/Marvel House Style).
    - Bold, expressive ink outlines, heavy comic-book cross-hatching and shading.
    - Saturated, dramatic color palettes or high-contrast B&W.
  `,
  DETAILED_COMIC: `
    ART STYLE: High-Detail Digital Graphic Novel.
    - Realistic anatomical rendering, cinematic lighting, painterly textures.
    - Soft, complex shadows and immersive environments.
  `
};

const MASTER_CONTINUITY_INSTRUCTION = `
ACT AS A WORLD-CLASS COMIC ARTIST AND DIRECTOR.

1. READABLE ENGLISH DIALOGUE (HIGHEST PRIORITY):
   - ANY TEXT IN SPEECH BUBBLES MUST BE CLEAR, LEGIBLE ENGLISH.
   - USE BOLD, BLOCK CAPITAL LETTERS (COMIC BOOK FONT).
   - ABSOLUTELY NO GIBBERISH, PSEUDO-TEXT, OR AI SYMBOLS.
   - IF YOU CANNOT RENDER THE TEXT PERFECTLY, LEAVE THE BUBBLE EMPTY.

2. CHARACTER DNA LOCK:
   - YOU MUST REPLICATE FACES, HAIR, AND CLOTHING SILHOUETTES EXACTLY ACROSS ALL PANELS.
   - USE THE PROVIDED CHARACTER REFERENCE AS THE SOURCE OF TRUTH.
   - NO OUTFIT CHANGES UNLESS SCRIPT EXPLICITLY STATES DAMAGE.

3. COLOR PURITY:
   - IF MODE IS 'B&W': USE ONLY BLACK, WHITE, AND GREY SCREENTONES. ZERO COLORS.
   - IF MODE IS 'COLOR': USE THE MEDIUM-SPECIFIC COLOR PALETTE (VIBRANT FOR DC, REALISTIC FOR DETAILED).

4. FIGHT CHOREOGRAPHY:
   - ATTACKS FOLLOW: 1) PREPARATION -> 2) INITIATION -> 3) IMPACT -> 4) AFTERMATH.
   - USE HIGH-IMPACT STYLE ONLY FOR STAGE 3.
`;

export const geminiService = {
  getAI() {
    return new GoogleGenAI({ apiKey: process.env.API_KEY });
  },

  async extractAssets(story: string, styleRef?: string): Promise<{ characters: Character[], locations: Location[] }> {
    const ai = this.getAI();
    const response = await ai.models.generateContent({
      model: "gemini-3-pro-preview",
      contents: `ACT AS A LEAD PRODUCTION DESIGNER. Analyze this story and extract 2-4 primary characters and key locations.
      Define a "Visual DNA" for each character (hair, eyes, permanent outfit).
      Script: ${story}
      Production Context: ${styleRef || 'Professional production standards'}`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            characters: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  name: { type: Type.STRING },
                  description: { type: Type.STRING },
                  visualPrompt: { type: Type.STRING },
                  visualAnchor: { type: Type.STRING }
                },
                required: ["name", "description", "visualPrompt", "visualAnchor"]
              }
            },
            locations: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  name: { type: Type.STRING },
                  description: { type: Type.STRING }
                },
                required: ["name", "description"]
              }
            }
          },
          required: ["characters", "locations"]
        }
      }
    });

    const data = JSON.parse(response.text || '{}');
    return {
      characters: (data.characters || []).map((c: any, i: number) => ({ ...c, id: `char-${i}-${Date.now()}` })),
      locations: (data.locations || []).map((l: any, i: number) => ({ ...l, id: `loc-${i}-${Date.now()}` }))
    };
  },

  async generateLocationPlates(location: Location, colorMode: ColorMode, medium: VisualMedium): Promise<string> {
    const ai = this.getAI();
    const modeStr = colorMode === 'B&W' ? 'STRICT PURE BLACK AND WHITE INK ONLY.' : 'FULL DIGITAL COLOR.';
    const styleRef = STYLE_PROTOCOLS[medium];
    
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-image',
      contents: { parts: [{ text: `ENVIRONMENT PLATE: ${location.name}. ${location.description}. ${modeStr} ${styleRef}. Cinematic composition. NO TEXT.` }] },
      config: { imageConfig: { aspectRatio: "16:9" } }
    });
    for (const part of response.candidates?.[0]?.content?.parts || []) {
      if (part.inlineData) return `data:image/png;base64,${part.inlineData.data}`;
    }
    return '';
  },

  async generateCharacterPortrait(character: Character, colorMode: ColorMode, medium: VisualMedium): Promise<string> {
    const ai = this.getAI();
    const modeStr = colorMode === 'B&W' ? 'STRICT PURE BLACK AND WHITE INK ONLY.' : 'FULL DIGITAL COLOR.';
    const styleRef = STYLE_PROTOCOLS[medium];
    
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-image',
      contents: { parts: [{ text: `CHARACTER MODEL SHEET: ${character.name}. DNA: ${character.visualAnchor}. ${modeStr} ${styleRef}. Front-facing neutral pose. NO TEXT.` }] },
      config: { imageConfig: { aspectRatio: "1:1" } }
    });
    for (const part of response.candidates?.[0]?.content?.parts || []) {
      if (part.inlineData) return `data:image/png;base64,${part.inlineData.data}`;
    }
    return '';
  },

  async generateCover(project: MangaProject): Promise<string> {
    const ai = this.getAI();
    const leadChar = project.characters[0];
    const modeStr = project.colorMode === 'B&W' ? 'STRICT PURE BLACK AND WHITE INK ONLY.' : 'FULL DIGITAL COLOR.';
    const styleRef = STYLE_PROTOCOLS[project.visualMedium];
    
    const parts: any[] = [];
    if (leadChar?.imageUrl) {
      parts.push({ text: `REFERENCE PROTAGONIST DNA:` }, { inlineData: { data: leadChar.imageUrl.split(',')[1], mimeType: 'image/png' } });
    }

    const prompt = `
      OFFICIAL COMIC COVER: "${project.title}".
      FEATURING: ${leadChar?.name}. DNA Lock: ${leadChar?.visualAnchor}.
      ${modeStr} ${styleRef}.
      MANDATORY: Render the title "${project.title}" in clear, stylized English font. NO GIBBERISH.
    `;
    parts.push({ text: prompt });

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-image',
      contents: { parts },
      config: { imageConfig: { aspectRatio: "3:4" } }
    });
    for (const part of response.candidates?.[0]?.content?.parts || []) {
      if (part.inlineData) return `data:image/png;base64,${part.inlineData.data}`;
    }
    return '';
  },

  async createStoryboard(project: MangaProject): Promise<StoryboardPage[]> {
    const ai = this.getAI();
    const response = await ai.models.generateContent({
      model: "gemini-3-pro-preview",
      contents: `ACT AS A SENIOR COMIC DIRECTOR. Create a ${project.pageCount}-page storyboard for "${project.title}".
      Medium: ${project.visualMedium}. Script: ${project.story}`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              pageNumber: { type: Type.INTEGER },
              layoutType: { type: Type.STRING, enum: ['SPLASH', 'ACTION_PANELS', 'NARRATIVE_PANELS', 'CLIMAX_SPREAD'] },
              layoutDescription: { type: Type.STRING },
              narrativeText: { type: Type.STRING },
              visualPrompt: { type: Type.STRING },
              charactersInPage: { type: Type.ARRAY, items: { type: Type.STRING } },
              locationId: { type: Type.STRING }
            },
            required: ["pageNumber", "layoutType", "layoutDescription", "narrativeText", "visualPrompt", "charactersInPage", "locationId"]
          }
        }
      }
    });
    
    try {
      const rawPages = JSON.parse(response.text || '[]');
      return rawPages.map((page: any) => ({
        ...page,
        status: 'pending',
        generatedImageUrl: undefined
      }));
    } catch (e) {
      console.error("Storyboard Parse Error:", e, response.text);
      throw new Error("Manga engine failed to parse storyboard. Please refine your script and try again.");
    }
  },

  async generateMangaPage(page: StoryboardPage, project: MangaProject): Promise<string> {
    const ai = this.getAI();
    const location = project.locations.find(l => l.id === page.locationId);
    const chars = project.characters.filter(c => page.charactersInPage.includes(c.name));
    
    const parts: any[] = [];
    if (location?.imageUrl) {
      parts.push({ text: `ENVIRONMENT CONTEXT:` }, { inlineData: { data: location.imageUrl.split(',')[1], mimeType: 'image/png' } });
    }
    chars.forEach(c => {
      if (c.imageUrl) {
        parts.push({ text: `CHARACTER DNA LOCK (${c.name}):` }, { inlineData: { data: c.imageUrl.split(',')[1], mimeType: 'image/png' } });
      }
    });

    const styleRef = STYLE_PROTOCOLS[project.visualMedium];
    const colorInstruction = project.colorMode === 'B&W' ? 'STRICT PURE BLACK AND WHITE INK ONLY. NO COLORS.' : 'FULL DIGITAL COLOR RENDERING.';

    const promptText = `
      ${MASTER_CONTINUITY_INSTRUCTION}
      GENERATING PAGE ${page.pageNumber}.
      ${styleRef}
      ${colorInstruction}
      SCENE DESCRIPTION: ${page.visualPrompt}
      DIALOGUE TO RENDER: "${page.narrativeText}"
      MANDATORY: PUT "${page.narrativeText}" IN A WHITE SPEECH BUBBLE WITH BOLD BLACK TEXT. 
      THE TEXT MUST BE CLEAR READABLE ENGLISH. ABSOLUTELY NO GIBBERISH.
      MAINTAIN CHARACTER IDENTITY FROM REFERENCES PERFECTLY.
    `;
    parts.push({ text: promptText });

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-image',
      contents: { parts },
      config: { imageConfig: { aspectRatio: "3:4" } }
    });

    for (const part of response.candidates?.[0]?.content?.parts || []) {
      if (part.inlineData) return `data:image/png;base64,${part.inlineData.data}`;
    }
    return '';
  }
};
