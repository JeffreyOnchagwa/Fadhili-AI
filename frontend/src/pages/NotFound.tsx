import { Link } from "react-router-dom";
import { Button } from "../components/ui/Button";

export default function NotFound() {
  return (
    <div className="mx-auto flex max-w-xl flex-col items-center gap-4 px-5 py-24 text-center">
      <h1 className="text-3xl font-medium">Page not found</h1>
      <p className="text-ink-soft">
        The page you're looking for doesn't exist or has moved.
      </p>
      <Link to="/">
        <Button>Back to home</Button>
      </Link>
    </div>
  );
}
