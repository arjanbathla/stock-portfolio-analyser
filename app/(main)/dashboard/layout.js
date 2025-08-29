import DashboardPage from "./page";
import { BarLoader } from "react-spinners";
import { Suspense } from "react";

export default function Layout() {
  return (
    <Suspense fallback={<BarLoader />}>
      <DashboardPage />
    </Suspense>
  );
};

