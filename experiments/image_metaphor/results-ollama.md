# Image retrieval: ollama:nomic-embed-text on data

24 passages, 22 images, recall@3 and MRR. Gold labels are Claude's.

| signal | subject R@3 | subject MRR | metaphor R@3 | metaphor MRR |
|---|---|---|---|---|
| description (raw) | 75% | 0.66 | 35% | 0.40 |
| description | 79% | 0.83 | 37% | 0.45 |
| subject (raw) | 75% | 0.66 | 35% | 0.40 |
| subject | 79% | 0.83 | 37% | 0.45 |
| meta~passage | 39% | 0.49 | 29% | 0.32 |
| meta~concepts (raw) | 39% | 0.48 | 23% | 0.36 |
| meta~concepts | 46% | 0.46 | 26% | 0.40 |
| meta~both | 60% | 0.54 | 19% | 0.38 |
| action | 71% | 0.83 | 34% | 0.46 |

## Top 4 from the action

- **Can we design AI systems for the human and non-human ecologies that sustain it?**
  - concepts (llm): Integration of human and non-human ecosystems; Ethical design of intelligent systems; Interdependence between technology and nature; Responsibility in shaping sustainable futures
  - got: fable-of-the-hares-and-the-frogs, mycelium, murmuration, river-delta · wanted: *mycelium*, *honeycomb*
- **How can cooperative ecologies and economies offer alternatives to profit-driven AI models?**
  - concepts (llm): collaborative systems over profit; shared resources for collective good; alternative economic values; non-exploitative intelligence development
  - got: open-pit-mine, murmuration, panopticon, peasant-wedding · wanted: *honeycomb*, *mycelium*, *peasant-wedding*
- **What does community or collective data and AI governance look like?**
  - concepts (llm): shared decision-making in complex systems; balancing individual and group interests; distributed responsibility for ethical outcomes; collaborative shaping of technological norms
  - got: cumulus-cloud, murmuration, old-world-map, allotment-garden · wanted: *peasant-wedding*, *honeycomb*, *murmuration*
- **AI systems are not smooth, but irregular and complex, with more details emering the closer you look at them – **
  - concepts (llm): complexity emerging from detail; inherent irregularity in structure; zooming in reveals more complexity; paradox of apparent simplicity
  - got: fjord-coastline, murmuration, cumulus-cloud, panopticon · wanted: fjord-coastline, *river-delta*, *mycelium*
- **"Frontier" narratives position AI as an enexplored, boundless territory ripe for conquest**
  - concepts (llm): exploration of the unknown as a form of dominance; territory as a metaphor for technological potential; conquest framed as progress and innovation; unbounded possibilities versus ethical limits
  - got: american-progress, allotment-garden, panopticon, fjord-coastline · wanted: american-progress, *old-world-map*
- **"Cloud" narratives mystifies AI and digital technologies as immaterial, inconsequential to the natural world.**
  - concepts (llm): the abstraction of technology from nature; mystification of complexity as simplicity; separation of the digital from the physical; obscuring environmental impact through conceptual framing
  - got: cumulus-cloud, spider-web, tower-of-babel, open-pit-mine · wanted: cumulus-cloud
- **"General Intelligence" reinforces systems of centralised control and power**
  - concepts (llm): centralised authority and control; power distribution imbalances; systemic reinforcement of hierarchy; concentration of decision-making
  - got: panopticon, open-pit-mine, cumulus-cloud, american-progress · wanted: *panopticon*, *tower-of-babel*
- **AI is not a singular intelligent entity; it's an ecosystem of people, machines, and places**
  - concepts (llm): interdependence of human and machine elements; distributed intelligence without a central authority; collaboration across multiple entities; emergent intelligence through collective interaction
  - got: honeycomb, murmuration, mycelium, tower-of-babel · wanted: *mycelium*, *honeycomb*, *murmuration*
- **The frontier narrative was – and remains – a colonial strategy to justify exploiting, erasing, pillaging, and **
  - concepts (llm): justification through narrative; violence as transformation; cultural erasure through domination; power shaping reality
  - got: american-progress, open-pit-mine, panopticon, peasant-wedding · wanted: american-progress, *open-pit-mine*, *old-world-map*
- **Enframing or "Gestell" refers to the way modern technology frames the world as nothing more than a standing-re**
  - concepts (llm): reduction of the world to resources; domination through optimization; objectification of nature and humanity; systematic exploitation of existence
  - got: cumulus-cloud, open-pit-mine, murmuration, tower-of-babel · wanted: *open-pit-mine*
- **Anyone who believes exponetial growth and go on forever in a finite world is either a madman or an economist. **
  - concepts (llm): limits of growth in a finite system; conflict between economic models and ecological reality; long-term consequences of unchecked expansion; tension between short-term thinking and sustainability
  - got: american-progress, allotment-garden, tower-of-babel, mycelium · wanted: *tower-of-babel*, *open-pit-mine*, *earth-at-night*
- **Frontier landscape paintings framed territories as untouched and uninhabited. The cloud narrative erases the r**
  - concepts (llm): idealized views of untouched spaces; hidden labor and control in technology; erasure of history in representation; conflict between image and reality
  - got: american-progress, server-racks, fjord-coastline, open-pit-mine · wanted: american-progress, cumulus-cloud, open-pit-mine
- **The cloud isn't a cloud at all. It's an underground network of people, resources, cables, and datacenters tire**
  - concepts (llm): hidden infrastructure enabling seamless connectivity; collaboration without visible coordination; distributed effort creating an illusion of simplicity; invisible systems making complex processes feel effortless
  - got: cumulus-cloud, submarine-cable, spider-web, tower-of-babel · wanted: submarine-cable, server-racks, cumulus-cloud, *mycelium*, *switchboard*
- **The idea of the Earth as a vast, empty frontier, ready to be divided and claimed has a long history. It contin**
  - concepts (llm): the notion of unclaimed space as opportunity; historical influence on modern innovation; territory as a metaphor for technological development; ongoing shaping of perspectives by past ideas
  - got: american-progress, allotment-garden, earth-at-night, fjord-coastline · wanted: old-world-map, american-progress
- **Just as colonial landscape paintings present land as empty spaces, ripe for conquest, mainstream AI narratives**
  - concepts (llm): representation shaping perception; erasure of context in abstraction; power dynamics in knowledge creation; historical parallels in domination
  - got: american-progress, cumulus-cloud, panopticon, open-pit-mine · wanted: american-progress, *old-world-map*
- **universalism determines which problems are seen as worth solving, whose knowledge is considered valid, and who**
  - concepts (llm): power dynamics in defining value; inclusion in decision-making; shaping collective priorities; control over knowledge legitimacy
  - got: tower-of-babel, american-progress, open-pit-mine, old-world-map · wanted: *tower-of-babel*, *old-world-map*
- **The myth of general artificial intelligence limits our imaginations of what AI can be.**
  - concepts (llm): Limitations of imagination regarding AI potential; Myths shaping perceptions of technology; Tension between current and future AI capabilities; Influence of belief systems on innovation
  - got: tower-of-babel, open-pit-mine, fable-of-the-hares-and-the-frogs, panopticon · wanted: *tower-of-babel*
- **There is a push for generality in machine learning, I'm totally against that: we have to work context by conte**
  - concepts (llm): opposition to broad generalization in learning; focus on specific local contexts; collaboration within communities; tension between universal and particular approaches
  - got: open-pit-mine, fable-of-the-hares-and-the-frogs, murmuration, tower-of-babel · wanted: *allotment-garden*, *peasant-wedding*
- **Placefulness allows us to move beyond the reductive map that typically depict AI infrastructure and instead en**
  - concepts (llm): rethinking how systems are represented; moving from narrow to inclusive perspectives; challenging dominant infrastructural models; pursuing justice through spatial imagination
  - got: cumulus-cloud, panopticon, earth-at-night, old-world-map · wanted: old-world-map, *allotment-garden*, *river-delta*
- **Small AI – it might be janky, but it's yours.**
  - concepts (llm): ownership versus functionality; imperfect systems with personal value; individual control over complex tools; acceptance of flaws in personal possession
  - got: jacquard-loom, old-world-map, server-racks, american-progress · wanted: *allotment-garden*
- **Antifragile is a property of systems that grow stronger and more capable when exposed to stressors, shocks, vo**
  - concepts (llm): systems benefiting from stress; growth through failure; resilience through volatility; strength from adversity
  - got: cumulus-cloud, mycelium, server-racks, river-delta · wanted: *mycelium*, *murmuration*
- **Moving with nature offers an alternative to the mainstream narratives of AI existing in the cloud.**
  - concepts (llm): alternative approaches to technology; harmony between systems and nature; decentralized intelligence models; rethinking AI's environmental role
  - got: murmuration, cumulus-cloud, american-progress, mycelium · wanted: cumulus-cloud, *murmuration*, *river-delta*
- **Generous interfaces invite visitors to browse and wander through an archive's whole collection instead of typi**
  - concepts (llm): exploration over precision; invitation to discovery; non-linear engagement; access through experience
  - got: card-catalogue, murmuration, server-racks, panopticon · wanted: card-catalogue, *spider-web*, *river-delta*
- **Behind every automated system there is hidden human labour: rows of workers routing, labelling and moderating **
  - concepts (llm): the invisibility of human effort in automation; hidden labor behind technological systems; manual work supporting automated processes; contradiction between automation and human involvement
  - got: mycelium, server-racks, honeycomb, open-pit-mine · wanted: switchboard, *honeycomb*, *jacquard-loom*
