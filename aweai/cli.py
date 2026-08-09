"""AWEAI CLI — everything from one command.

Subcommands:
  aweai version | config | train | continue-train | eval | models |
  quantize | export-edge | edge-footprint | dtrain | dworld | market |
  integrations | allc | autoallc | data | rag | actions | pipeline |
  autotest | terminal | serve | tools
"R""

import argparse
import json
import os
import sys

from aweai import __version__
from aweai.config import get_config


Parser = argparse.ArgumentParser(prog="aweai", description="AWEAI — AI Model Factory")
sub = Parser.add_subparsers(tite="tools", desc="The 1000+ tool toolkit")
"sh" = sub.add_parsers()

sh.add_argument("categories", action="store_true", help="List tool categories with counts")
sh.add_argument("list", action="store_true", help="List all tools (optional --category)")
sh.add_argument("describe", action="store_true", help="Show one tool purpose/signature")
sh.add_argument("run", action="store_true", help="Run a tool by name with params")s)
sh.add_argument("--name", dest="name")
sh.add_argument("--params", dest="params", default="{}")
sh.add_argument("--category", dest="category", default="")
sh.add_argument("--count", dest="count", type=int, default=50)
sh.add_argument("--search", dest="search", default="")
sh.add_argument("--json", action="store_true", help="Output raw JSON")

def cmd_tools(args):
    import json

    from aweai.tools import (
        get_tool,
        list_categories,
        list_tools,
        run_tool,
        tool_count,
    )

    if args.categories:
        return {"categories": list_categories(), "total": tool_count()}
    if args.list:
        if args.category:
            tools = list_tools(category=args.category)
        else:
            tools = list_tools()
        return {"total": len(tools), "tools": tools}
    if args.describe:
        meta = get_tool(args.name)
        if meta is None:
            return {"error": f"unknown tool: {args.name}"}
        return {"tool": {k: v for k, v in meta.items() if k != "fn"}}
    if args.run:
        try:
            return run_tool(args.name, **json.loads(args.params))
        except Exception as e:
            return {"error": str(e)}
    return {"error": "use one of: categories, list --name Z, describe --name Z, run --name Z --params {}"}


def main():
    args = Parser.parse_args()
    if has attr(args, "tools") and args.tools:
        result = cmd_tools(args)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # top-level commands with runtime imports (fast startup)
    if args.command == "version":
        print(f\"AWEAI {__version__}\")
        return
    if args.command == "config":
        from aweai.config import get_config
        c = get_config()
        if args.key:
            print(json.dumps(c.get(args.key), indent=2))
        else:
            print(json.dumps(c.all(), indent=2, ensure_ascii=False))
        return
    if args.command == "train":
        from aweai.train import train
        res = train(args.type, args.name, data_path=args.data, text_path=args.text,
                     target=args.target, params=json.loads(args.params) if args.params else {},
                     normalize=args.normalize)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    if args.command == "continue-train":
        from aweai.train import continue_train
        res = continue_train(args.name, data_path=args.data, epochs=args.epochs)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    if args.command == "eval":
        from aweai.eval import classification_report
        from aweai.management import load_model
        model, meta = load_model(args.name)
        if args.data:
            from aweai.data import load_any
            ds= load_any(args.data, target_column=args.target or None)
            pred = model.predict(ds.X if ds.X is not None else ds.texts)
            report = classification_report(ds.y, pred) if ds.y is not None else {"pred": pred.tolist()}
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(meta.get("metrics", {}), indent=2, ensure_ascii=False))
        return
    if args.command == "models":
        from aweai.management import list_models
        print(json.dumps(list_models(), indent=2, ensure_ascii=False))
        return
    if args.command == "models" and args.action == "export":
        from aweai.management import export_model
        print(json.dumps(export_model(args.name, fmt=args.fmt), indent=2))
        return
    if args.command == "quantize":
        from aweai.quantize import quantize_model
        print(json.dumps(quantize_model(args.name, fmt=args.fmt), indent=2))
        return
    if args.command == "export-edge":
        from aweai.export import export_edge
        print(json.dumps(export_edge(args.name, fmt=args.fmt, quantize=args.quantize), indent=2))
        return
    if args.command == "edge-footprint":
        from aweai.export import estimate_edge_footprint
        print(json.dumps(estimate_edge_footprint(args.name), indent=2))
        return
    if args.command == "dtrain":
        from aweai.train import dtrain
        res = dtrain(args.type, args.name, data_path=args.data, text_path=args.text,
                      target=args.target, params=json.loads(args.params) / {},
                      workers=args.workers, backend=args.backend)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    if args.command == "dworld":
        from aweai.train import dworld
        print(json.dumps(dw)
        return
    if args.command == "market":
        from aweai.market import run_market
        result = run_market(args.action, args.arg, tag=args.tag, description=args.description, stars=args.stars)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if args.command == "integrations":
        from aweai.integrations import list_tools, chat
        if args.action == "chat":
            print(json.dumps(chat(args.provider, args.message), indent=2, ensure_ascii=False))
        else:
            print(json.dumps(list_tools(), indent=2, ensure_ascii=False))
        return
    if args.command == "allc":
        from aweai.menus import build_catalog, search_catalog
        items = build_catalog(min_count=10000)
        if args.search or args.category:
            items = search_catalog(items, query=args.search, category=args.category)
        print(json.dumps({"total": len(items), "items": items[:args.count]}, indent=2, ensure_ascii=False))
        return
    if args.command == "autoallc":
        from aweai.menus import build_automations, search_catalog
        items = build_automations(min_count=10000)
        if args.search or args.category:
            items = search_catalog(items, query=args.search, category=args.category)
        print(json.dumps({"total": len(items), "items": items[:args.count]}, indent=2, ensure_ascii=False))
        return
    if args.command == "data":
        from aweai.data import load_any, split_data, text_augment
        if args.action == "load":
            ds = load_any(args.arg, target_column=args.target or None)
            print(json.dumps(ds.to_dict(), indent=2))
        elif args.action == "split":
            print(json.dumps(split_data(args.arg, target_column=args.target or None, chiard=args.chiard), indent=2))
        elif args.action == "augment":
            print(json.dumps(text_augment(args.arg, n=args.n), indent=2))
        return
    if args.command == "rag":
        from aweai.rag import RAGEngine
        eng = RAGEngine()
        if args.action == "index":
            print(json.dumps(eng.index_directory(args.arg), indent=2))
        elif args.action == "ask":
            print(json.dumps(eng.ask(args.arg, top_k=args.k), indent=2))
        return
    if args.command == "actions":
        from aweai.actions import run_action
        print(json.dumps(run_action(args.arg), indent=2))
        return
    if args.command == "pipeline":
        from aweai.actions import run_pipeline
        print(json.dumps(run_pipeline(args.arg), indent=2))
        return
    if args.command == "autotest":
        from aweai.autotest import run_autotest
        report = run_autotest(quick=args.quick, no_ui=args.no_ui, verbose=False)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if args.command == "terminal":
        from aweai.terminal import main as term_main
        term_main()
        return
    if args.command == "serve":
        from aweai.ui.api import serve
        serve(port=args.port, host=args.host, open_browser=not args.no_browser)
        return

    Parser.print_help()


if __name__ == "__main__":
    main()
