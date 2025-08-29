"use client";

import { useEffect } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { CalendarIcon, Loader2 } from "lucide-react";
import { format } from "date-fns";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
} from "@/components/ui/popover";
import { Calendar } from "@/components/ui/calendar";
import { cn } from "@/lib/utils";
import useFetch from "@/hooks/use-fetch";
import { createStock } from "@/actions/stocks"; // 👈 You'll write this

// Zod schema
const stockSchema = z.object({
  ticker: z.string().min(1, "Ticker is required"),
  shares: z.coerce.number().positive("Shares must be greater than 0"),
  purchaseDate: z.date(),
  purchasePrice: z.coerce.number().optional(),
  targetWeight: z.coerce.number().optional(),
  reinvestDividends: z.boolean().optional(),
});

export function AddStockForm({ portfolioId }) {
  const router = useRouter();

  const {
    register,
    handleSubmit,
    formState: { errors },
    setValue,
    watch,
    reset,
  } = useForm({
    resolver: zodResolver(stockSchema),
    defaultValues: {
      ticker: "",
      shares: "",
      purchaseDate: new Date(),
      purchasePrice: "",
      targetWeight: "",
      reinvestDividends: false,
    },
  });

  const { fn: submitStock, loading, data: result } = useFetch(createStock);
  const date = watch("purchaseDate");

  const onSubmit = (formData) => {
    submitStock({ ...formData, portfolioId });
  };

  useEffect(() => {
    if (result?.success && !loading) {
      toast.success("Stock added successfully!");
      reset();
      router.push(`/dashboard/`);
    }
  }, [result, loading]);

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
      {/* Ticker */}
      <div className="space-y-2">
        <label className="text-sm font-medium">Ticker</label>
        <Input placeholder="AAPL" {...register("ticker")} />
        {errors.ticker && <p className="text-sm text-red-500">{errors.ticker.message}</p>}
      </div>

      {/* Shares */}
      <div className="space-y-2">
        <label className="text-sm font-medium">Shares</label>
        <Input type="number" step="0.01" placeholder="e.g. 10" {...register("shares")} />
        {errors.shares && <p className="text-sm text-red-500">{errors.shares.message}</p>}
      </div>

      {/* Purchase Date */}
      <div className="space-y-2">
        <label className="text-sm font-medium">Purchase Date</label>
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" className={cn("w-full text-left", !date && "text-muted-foreground")}>
              {date ? format(date, "PPP") : "Pick a date"}
              <CalendarIcon className="ml-auto h-4 w-4 opacity-50" />
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-auto p-0">
            <Calendar
              mode="single"
              selected={date}
              onSelect={(d) => setValue("purchaseDate", d)}
              disabled={(d) => d > new Date()}
              initialFocus
            />
          </PopoverContent>
        </Popover>
        {errors.purchaseDate && <p className="text-sm text-red-500">{errors.purchaseDate.message}</p>}
      </div>

      {/* Purchase Price */}
      <div className="space-y-2">
        <label className="text-sm font-medium">Purchase Price (optional)</label>
        <Input type="number" step="0.01" placeholder="e.g. 150.25" {...register("purchasePrice")} />
        {errors.purchasePrice && <p className="text-sm text-red-500">{errors.purchasePrice.message}</p>}
      </div>

      {/* Target Weight */}
      <div className="space-y-2">
        <label className="text-sm font-medium">Target Weight % (optional)</label>
        <Input type="number" step="0.1" placeholder="e.g. 10" {...register("targetWeight")} />
        {errors.targetWeight && <p className="text-sm text-red-500">{errors.targetWeight.message}</p>}
      </div>

      {/* Reinvest Dividends */}
      <div className="flex items-center justify-between rounded-lg border p-4">
        <div>
          <label className="text-base font-medium">Reinvest Dividends</label>
          <p className="text-sm text-muted-foreground">Automatically reinvest dividends</p>
        </div>
        <Switch onCheckedChange={(checked) => setValue("reinvestDividends", checked)} />
      </div>

      {/* Actions */}
      <div className="flex justify-center gap-4">
        <Button type="button" variant="outline" className="w-32" onClick={() => router.back()}>
          Cancel
        </Button>
        <Button type="submit" className="w-32" disabled={loading}>
          {loading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Adding...
            </>
          ) : (
            "Add Stock"
          )}
        </Button>
      </div>
    </form>
  );
}

