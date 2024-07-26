* Integrate Fairness-aware methods (update branch, merge) (2 weeks)

* Add Analytics (API + processes + SecAgg (incl. Metrics)) (1 Month)


* Refactor routines
  - NOTES:
    - Write some structure to handle Model + Optimizer(s) + Aggregator
      - Write logic for setup (duplicate from Server to Client / Peer to Peer)
      - Write logic for local training (provided data is available)
      - Write logic for aggregation (then, modularize to support Gossip, etc.)

* Revise serialization (move to Msgpack ; revise custom code ; screen perfs) (1 week)


---


* Configuration tools => improve usability and extensibility ; see with Rosalie's work

* Interface and use FLamby (for examples and/or benchmarks)
  -> See with Paul & Edwige

* Nouveaux algos
  - Personalization via Hybrid training (2 weeks)
  - Things with Rosalie?

* Profile performances (benchmark: with asv or, easier, using current logging)

* Revise Network Communication:
    - Modularize timeout on responses => go minimal on that
    - Enable connection loss/re-connection => not now, wait for tests / actual problems
    - Improve the way clients are identified by MessagesHandler? => test again for issues (see if required/interesting)
    - Improve the tackling of MessagesHandler receiving multiple messages from or for the same client? => wait for roadmap on decentralized

* Add client sampling

* Split NetworkServer from FederatedServer

* (Later) Quickrun mode revision
