import { getUserPortfolios } from "@/actions/dashboard"; // ✅ You should implement or already have this
import { AddStockForm } from "../../transaction/_components/AddStockForm"; // ✅ Adjust this path as needed

export default async function AddStockPage() {
  const portfolios = await getUserPortfolios(); // assumes a list of { id, name }

  // Use default portfolio if only one, or let the user pick in the form
  const defaultPortfolio = portfolios[0];

  return (
    <div className="max-w-3xl mx-auto px-5">
      <div className="flex justify-center md:justify-normal mb-8 pt-8">
        <h1 className="text-5xl gradient-title">Add Stock</h1>
      </div>

      {/* Pass default portfolio ID to form (or modify form to select it) */}
      <AddStockForm portfolioId={defaultPortfolio?.id} />
    </div>
  );
} 