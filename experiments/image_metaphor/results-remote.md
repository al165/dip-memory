# Image retrieval: remote:google/embeddinggemma-2 on data-embeddinggemma2

24 passages, 22 images, recall@3 and MRR. Gold labels are Claude's.

| signal | subject R@3 | subject MRR | metaphor R@3 | metaphor MRR |
|---|---|---|---|---|
| visual | 93% | 0.88 | 31% | 0.45 |
| description (raw) | 82% | 0.89 | 33% | 0.51 |
| description | 90% | 0.90 | 36% | 0.48 |
| subject (raw) | 94% | 0.83 | 28% | 0.47 |
| subject | 90% | 0.92 | 29% | 0.49 |
| meta~passage | 65% | 0.53 | 32% | 0.43 |
| meta~concepts (raw) | 31% | 0.44 | 34% | 0.46 |
| meta~concepts | 31% | 0.43 | 42% | 0.51 |
| meta~both | 31% | 0.43 | 42% | 0.51 |
| action | 90% | 0.89 | 38% | 0.50 |

## Top 4 from the action

- **Can we design AI systems for the human and non-human ecologies that sustain it?**
  - concepts (llm): integration of human and non-human systems; ethical responsibility in AI design; interdependence between technology and ecology; balancing innovation with sustainability
  - got: honeycomb, mycelium, fable-of-the-hares-and-the-frogs, jacquard-loom · wanted: *mycelium*, *honeycomb*
- **How can cooperative ecologies and economies offer alternatives to profit-driven AI models?**
  - concepts (llm): collaborative systems over profit; shared resources for collective good; AI guided by communal values; alternatives to competitive models
  - got: honeycomb, allotment-garden, fable-of-the-hares-and-the-frogs, murmuration · wanted: *honeycomb*, *mycelium*, *peasant-wedding*
- **What does community or collective data and AI governance look like?**
  - concepts (llm): shared decision-making in data use; balancing individual and group interests; distributed responsibility for AI impact; collaborative shaping of ethical norms
  - got: honeycomb, murmuration, earth-at-night, mycelium · wanted: *peasant-wedding*, *honeycomb*, *murmuration*
- **AI systems are not smooth, but irregular and complex, with more details emering the closer you look at them – **
  - concepts (llm): complexity emerging from detail; inherent irregularity in structure; zooming in reveals more complexity; paradox of increasing detail
  - got: fjord-coastline, murmuration, river-delta, allotment-garden · wanted: fjord-coastline, *river-delta*, *mycelium*
- **"Frontier" narratives position AI as an enexplored, boundless territory ripe for conquest**
  - concepts (llm): AI as an uncharted domain for exploration; territory viewed as a resource to be claimed; expansion driven by ambition and discovery; tension between boundlessness and control
  - got: american-progress, server-racks, fjord-coastline, open-pit-mine · wanted: american-progress, *old-world-map*
- **"Cloud" narratives mystifies AI and digital technologies as immaterial, inconsequential to the natural world.**
  - concepts (llm): the abstraction of technology from nature; the illusion of detachment from the physical world; mystification of complex systems as simple; the tension between the material and the immaterial
  - got: cumulus-cloud, server-racks, tower-of-babel, murmuration · wanted: cumulus-cloud
- **"General Intelligence" reinforces systems of centralised control and power**
  - concepts (llm): centralised control reinforcing power structures; concentration of authority and influence; systemic reinforcement of hierarchical dominance; entrenchment of power through organisational design
  - got: panopticon, open-pit-mine, submarine-cable, tower-of-babel · wanted: *panopticon*, *tower-of-babel*
- **AI is not a singular intelligent entity; it's an ecosystem of people, machines, and places**
  - concepts (llm): interconnected systems working together; distributed intelligence without a single center; collaboration between human and machine elements; emergent complexity from collective interaction
  - got: cumulus-cloud, murmuration, earth-at-night, honeycomb · wanted: *mycelium*, *honeycomb*, *murmuration*
- **The frontier narrative was – and remains – a colonial strategy to justify exploiting, erasing, pillaging, and **
  - concepts (llm): justification through narrative; colonial domination and control; systematic erasure and violence; powerful ideologies shaping reality
  - got: american-progress, open-pit-mine, tower-of-babel, old-world-map · wanted: american-progress, *open-pit-mine*, *old-world-map*
- **Enframing or "Gestell" refers to the way modern technology frames the world as nothing more than a standing-re**
  - concepts (llm): reduction of the world to resources; domination through optimization; loss of intrinsic value; systemic exploitation of existence
  - got: tower-of-babel, open-pit-mine, panopticon, american-progress · wanted: *open-pit-mine*
- **Anyone who believes exponetial growth and go on forever in a finite world is either a madman or an economist. **
  - concepts (llm): limits of growth in a finite system; conflict between economic models and ecological reality; critique of unchecked expansion; tension between short-term gain and long-term sustainability
  - got: tower-of-babel, open-pit-mine, american-progress, mycelium · wanted: *tower-of-babel*, *open-pit-mine*, *earth-at-night*
- **Frontier landscape paintings framed territories as untouched and uninhabited. The cloud narrative erases the r**
  - concepts (llm): romanticizing untouched landscapes; hiding labor and power dynamics; contrasting idealized vs. real histories; erasing complexity in technological development
  - got: american-progress, open-pit-mine, panopticon, peasant-wedding · wanted: american-progress, cumulus-cloud, open-pit-mine
- **The cloud isn't a cloud at all. It's an underground network of people, resources, cables, and datacenters tire**
  - concepts (llm): the illusion of simplicity masking complex coordination; distributed effort creating seamless experience; invisible infrastructure enabling visible convenience; collaboration without centralized control
  - got: submarine-cable, tower-of-babel, cumulus-cloud, server-racks · wanted: submarine-cable, server-racks, cumulus-cloud, *mycelium*, *switchboard*
- **The idea of the Earth as a vast, empty frontier, ready to be divided and claimed has a long history. It contin**
  - concepts (llm): historical views of land as unclaimed space; shaping narratives around progress; continuity of old ideas in modern contexts; innovation framed as expansion and control
  - got: american-progress, the-large-pool-of-bursa, earth-at-night, peasant-wedding · wanted: old-world-map, american-progress
- **Just as colonial landscape paintings present land as empty spaces, ripe for conquest, mainstream AI narratives**
  - concepts (llm): erasure of context in representation; power dynamics in knowledge creation; abstraction as a form of omission; historical parallels in visual and technological narratives
  - got: american-progress, open-pit-mine, tower-of-babel, fjord-coastline · wanted: american-progress, *old-world-map*
- **universalism determines which problems are seen as worth solving, whose knowledge is considered valid, and who**
  - concepts (llm): power over what problems are addressed; determination of valid knowledge; inclusion in technological shaping; unequal participation in decision-making
  - got: tower-of-babel, american-progress, earth-at-night, open-pit-mine · wanted: *tower-of-babel*, *old-world-map*
- **The myth of general artificial intelligence limits our imaginations of what AI can be.**
  - concepts (llm): limiting imagination through dominant narratives; the tension between myth and reality; the influence of perception on possibility; the role of belief in shaping potential
  - got: american-progress, cumulus-cloud, fjord-coastline, murmuration · wanted: *tower-of-babel*
- **There is a push for generality in machine learning, I'm totally against that: we have to work context by conte**
  - concepts (llm): conflict between universal approaches and localized efforts; importance of context-specific solutions; resistance to one-size-fits-all strategies; emphasis on community-based knowledge
  - got: honeycomb, card-catalogue, murmuration, allotment-garden · wanted: *allotment-garden*, *peasant-wedding*
- **Placefulness allows us to move beyond the reductive map that typically depict AI infrastructure and instead en**
  - concepts (llm): rethinking AI infrastructure through justice and sustainability; moving beyond simplistic and harmful representations; imagining more ethical and life-affirming systems; challenging dominant paradigms of mapping and control
  - got: earth-at-night, open-pit-mine, old-world-map, panopticon · wanted: old-world-map, *allotment-garden*, *river-delta*
- **Small AI – it might be janky, but it's yours.**
  - concepts (llm): ownership versus functionality; imperfect tools with personal value; acceptance of imperfection; individual control over complex systems
  - got: fable-of-the-hares-and-the-frogs, server-racks, murmuration, panopticon · wanted: *allotment-garden*
- **Antifragile is a property of systems that grow stronger and more capable when exposed to stressors, shocks, vo**
  - concepts (llm): systems benefiting from stress and challenges; growth through exposure to failure and uncertainty; resilience as a dynamic and adaptive process; strength emerging from instability and disorder
  - got: spider-web, mycelium, submarine-cable, murmuration · wanted: *mycelium*, *murmuration*
- **Moving with nature offers an alternative to the mainstream narratives of AI existing in the cloud.**
  - concepts (llm): alternative approaches to technological existence; harmony between artificial systems and natural processes; decentralized intelligence models; rethinking the role of technology in the environment
  - got: cumulus-cloud, mycelium, murmuration, allotment-garden · wanted: cumulus-cloud, *murmuration*, *river-delta*
- **Generous interfaces invite visitors to browse and wander through an archive's whole collection instead of typi**
  - concepts (llm): exploration over precision; invitation to discovery; fluid interaction with information; experience versus search
  - got: old-world-map, river-delta, card-catalogue, fable-of-the-hares-and-the-frogs · wanted: card-catalogue, *spider-web*, *river-delta*
- **Behind every automated system there is hidden human labour: rows of workers routing, labelling and moderating **
  - concepts (llm): hidden labor behind automation; human effort in machine systems; manual work within technological processes; discrepancy between visible and invisible labor
  - got: switchboard, server-racks, honeycomb, open-pit-mine · wanted: switchboard, *honeycomb*, *jacquard-loom*
