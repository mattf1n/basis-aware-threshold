import mauve, torch, numpy as np
import json, sys, functools, argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--out", type=argparse.FileType("w"), default=sys.stdout, help="output file")
    parser.add_argument(
        "-n", type=int, default=None, help="run MAUVE on first `n` examples"
    )
    parser.add_argument(
        "--q-file",
        "-q",
        type=argparse.FileType("r"),
        default=sys.stdin,
        help="generated text",
    )
    parser.add_argument(
        "--p-file",
        "-p",
        type=argparse.FileType("r"),
        default="data/webtext/valid_tokenized.jsonl",
        help="human text",
    )
    args = parser.parse_args()
    q_tokens = load_tokens(args.q_file)[: args.n]
    p_tokens = load_tokens(args.p_file, fmt="list")[
        : len(q_tokens) if args.n is None else args.n
    ]
    out = mauve.compute_mauve(
        p_tokens=p_tokens, q_tokens=q_tokens, verbose=True, device_id=args.device
    )
    json.dump({k: np.array(v).tolist() for k, v in vars(out).items()}, args.out)


def load_tokens(file, fmt="dict"):
    if fmt == "dict":
        return list(
            torch.unsqueeze(torch.LongTensor(json.loads(line)["generation"]), dim=0)
            for line in file
        )
    else:
        return list(
            torch.unsqueeze(torch.LongTensor(json.loads(line)), dim=0) for line in file
        )


if __name__ == "__main__":
    main()
