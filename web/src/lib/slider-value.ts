/** Base UI sliders may emit a number or a number[] depending on thumb count. */
export function sliderNumber(value: unknown, fallback = 0): number {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (Array.isArray(value)) {
    const first = value[0];
    if (typeof first === "number" && Number.isFinite(first)) return first;
  }
  return fallback;
}

export function sliderPair(value: unknown, fallback: [number, number]): [number, number] {
  if (Array.isArray(value) && value.length >= 2) {
    const a = sliderNumber(value[0], fallback[0]);
    const b = sliderNumber(value[1], fallback[1]);
    return [a, b];
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return [value, fallback[1]];
  }
  return fallback;
}
