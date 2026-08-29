"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";

const LABELS: Record<string, string> = {
  finishing: "Finishing",
  creation: "Creation",
  passing: "Passing",
  carrying: "Carrying",
  occupation: "Pitch occupation",
  defending: "Defending",
  duels: "Duels",
  shotstopping: "Shot-stopping",
  distribution: "Distribution",
  sweeping: "Sweeping",
};

export function WeightSliders({
  weights,
  onChange,
}: {
  weights: Record<string, number>;
  onChange: (next: Record<string, number>) => void;
}) {
  const keys = Object.keys(weights);
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Metric weights</CardTitle>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            const reset: Record<string, number> = {};
            for (const k of keys) reset[k] = 1;
            onChange(reset);
          }}
        >
          Reset
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          Re-rank live. Turn defending up to find destroyers; turn carrying up to find dribblers.
        </p>
        {keys.map((key) => (
          <div key={key}>
            <Label className="mb-1 flex justify-between">
              <span>{LABELS[key] ?? key}</span>
              <span className="tabular-nums text-muted-foreground">{weights[key].toFixed(1)}×</span>
            </Label>
            <Slider
              min={0}
              max={3}
              step={0.1}
              value={[weights[key]]}
              onValueChange={(v) => onChange({ ...weights, [key]: v[0] })}
            />
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
