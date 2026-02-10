
export type ColorMode = 'B&W' | 'Color';

export type VisualMedium = 'MANGA' | 'DC_COMIC' | 'DETAILED_COMIC';

export type PageLayoutType = 'SPLASH' | 'ACTION_PANELS' | 'NARRATIVE_PANELS' | 'CLIMAX_SPREAD';

export interface Location {
  id: string;
  name: string;
  description: string;
  imageUrl?: string;
}

export interface Character {
  id: string;
  name: string;
  description: string;
  visualPrompt: string;
  visualAnchor: string; 
  imageUrl?: string;
}

export interface StoryboardPage {
  pageNumber: number;
  layoutType: PageLayoutType;
  layoutDescription: string;
  narrativeText: string;
  visualPrompt: string;
  charactersInPage: string[];
  locationId?: string; 
  generatedImageUrl?: string;
  status: 'pending' | 'generating' | 'completed' | 'failed';
}

export interface MangaProject {
  title: string;
  volumeName?: string;
  artistName: string;
  story: string;
  colorMode: ColorMode;
  visualMedium: VisualMedium;
  pageCount: number;
  characters: Character[];
  locations: Location[];
  storyboard: StoryboardPage[];
  styleReferences?: string;
  coverImageUrl?: string;
  coverStatus: 'pending' | 'generating' | 'completed' | 'failed';
}

export type AppStep = 'setup' | 'assets' | 'storyboard' | 'generation';
