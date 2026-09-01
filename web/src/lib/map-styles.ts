export const STYLE_COLORS: Record<string, { fill: string; ink: string }> = {
  "Deep-lying playmaker": { fill: "#04F5FF", ink: "#16001A" },
  "Box-to-box midfielder": { fill: "#00C853", ink: "#16001A" },
  Destroyer: { fill: "#37003C", ink: "#ffffff" },
  "Ball-playing centre-back": { fill: "#963CFF", ink: "#ffffff" },
  Stopper: { fill: "#5C4B60", ink: "#ffffff" },
  "Overlapping full-back": { fill: "#00FF85", ink: "#16001A" },
  "Inverted full-back": { fill: "#1B998B", ink: "#ffffff" },
  "Wing-back": { fill: "#7CFFB2", ink: "#16001A" },
  "Wide creator": { fill: "#FF2882", ink: "#ffffff" },
  "Inside forward": { fill: "#E90052", ink: "#ffffff" },
  Winger: { fill: "#FF8A00", ink: "#16001A" },
  "Target striker": { fill: "#16001A", ink: "#ffffff" },
  Poacher: { fill: "#FF5C8A", ink: "#16001A" },
  "False nine": { fill: "#C77DFF", ink: "#16001A" },
  "Shadow striker": { fill: "#FF6B35", ink: "#16001A" },
  "Pressing forward": { fill: "#2EC4B6", ink: "#16001A" },
};

export function styleColor(name: string): { fill: string; ink: string } {
  return STYLE_COLORS[name] ?? { fill: "#37003C", ink: "#ffffff" };
}
