import requests

TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

class SolanaRPC:
    def __init__(self, url, timeout=15):
        self.url=url
        self.timeout=timeout

    def call(self, method, params):
        payload={"jsonrpc":"2.0","id":1,"method":method,"params":params}
        r=requests.post(self.url,json=payload,timeout=self.timeout)
        r.raise_for_status()
        data=r.json()
        if "error" in data:
            raise RuntimeError(data["error"])
        return data["result"]

    def get_account_info(self, pubkey):
        return self.call("getAccountInfo",[pubkey,{"encoding":"jsonParsed","commitment":"confirmed"}])

    def get_token_supply(self, mint):
        return self.call("getTokenSupply",[mint,{"commitment":"confirmed"}])

    def get_largest_accounts(self, mint):
        return self.call("getTokenLargestAccounts",[mint,{"commitment":"confirmed"}])

def enrich_token_from_rpc(rpc, mint, label="TOKEN"):
    account=rpc.get_account_info(mint)
    supply=rpc.get_token_supply(mint)
    largest=rpc.get_largest_accounts(mint)

    value=account.get("value") if isinstance(account,dict) else None
    parsed={}
    owner_program=None
    if value:
        owner_program=value.get("owner")
        data=value.get("data")
        if isinstance(data,dict):
            parsed=data.get("parsed",{}).get("info",{})

    mint_authority=parsed.get("mintAuthority")
    freeze_authority=parsed.get("freezeAuthority")
    decimals=supply.get("value",{}).get("decimals")
    ui_supply=supply.get("value",{}).get("uiAmountString")

    largest_rows=(largest.get("value") or [])
    largest_amounts=[]
    for row in largest_rows:
        try:
            largest_amounts.append(float(row.get("uiAmount") or 0))
        except Exception:
            pass

    # Only the authority fields and raw largest-account observations are populated.
    # LP/liquidity/cluster/sellability are deliberately left unknown.
    return {
        "token": label,
        "mint": mint,
        "liquidity_usd": None,
        "lp_locked": None,
        "lp_lock_days": None,
        "mint_authority_active": bool(mint_authority),
        "freeze_authority_active": bool(freeze_authority),
        "wash_trade_risk": "UNKNOWN",
        "bundle_risk": "UNKNOWN",
        "sellable": None,
        "top10_holder_pct": None,
        "dev_holder_pct": None,
        "cluster_holder_pct": None,
        "dev_risk": "UNKNOWN",
        "contract_risk": "CLEAN" if owner_program == TOKEN_PROGRAM else "UNKNOWN",
        "decimals": decimals,
        "total_supply": ui_supply,
        "largest_accounts_raw": largest_amounts,
    }
