# Image retrieval: st:google/embeddinggemma-2 on data-embeddinggemma2

24 passages, 22 images, recall@3 and MRR. Gold labels are Claude's.

| signal | subject R@3 | subject MRR | metaphor R@3 | metaphor MRR |
|---|---|---|---|---|
| visual | 93% | 0.88 | 31% | 0.45 |
| description (raw) | 82% | 0.89 | 33% | 0.51 |
| description | 90% | 0.90 | 36% | 0.48 |
| subject (raw) | 94% | 0.83 | 28% | 0.44 |
| subject | 90% | 0.92 | 29% | 0.48 |
| meta~passage | 65% | 0.57 | 32% | 0.43 |
| meta~concepts (raw) | 43% | 0.47 | 25% | 0.41 |
| meta~concepts | 35% | 0.43 | 27% | 0.43 |
| meta~both | 35% | 0.43 | 27% | 0.43 |
| action | 90% | 0.90 | 32% | 0.48 |

## Top 4 from the action

- **Can we design AI systems for the human and non-human ecologies that sustain it?**
  - concepts (llm): symbiotic relationships between AI and ecosystems; ethical responsibility in technological development; interdependence of human and non-human life; designing systems that support sustainability
  - got: honeycomb, allotment-garden, fable-of-the-hares-and-the-frogs, mycelium · wanted: *mycelium*, *honeycomb*
- **How can cooperative ecologies and economies offer alternatives to profit-driven AI models?**
  - concepts (llm): collaborative systems beyond market control; shared resources guiding technological development; decentralized knowledge creation and distribution; ethical priorities shaping artificial intelligence
  - got: honeycomb, mycelium, allotment-garden, fable-of-the-hares-and-the-frogs · wanted: *honeycomb*, *mycelium*, *peasant-wedding*
- **What does community or collective data and AI governance look like?**
  - concepts (llm): shared responsibility for data control; collaborative decision-making without hierarchy; balancing innovation with ethical oversight; dynamic trust between groups and systems
  - got: honeycomb, fable-of-the-hares-and-the-frogs, earth-at-night, tower-of-babel · wanted: *peasant-wedding*, *honeycomb*, *murmuration*
- **AI systems are not smooth, but irregular and complex, with more details emering the closer you look at them – **
  - concepts (llm): complexity emerging through closer inspection; interconnected yet unpredictable patterns; depth revealing hidden intricacies; non-linear development of understanding
  - got: fjord-coastline, murmuration, river-delta, jacquard-loom · wanted: fjord-coastline, *river-delta*, *mycelium*
- **"Frontier" narratives position AI as an enexplored, boundless territory ripe for conquest**
  - concepts (llm): exploration of unknown potential; competition for control and dominance; human ambition meeting new frontiers; risk and reward in uncharted domains
  - got: american-progress, open-pit-mine, fjord-coastline, old-world-map · wanted: american-progress, *old-world-map*
- **"Cloud" narratives mystifies AI and digital technologies as immaterial, inconsequential to the natural world.**
  - concepts (llm): abstracting technology from physical reality; disconnection between digital and natural worlds; idealization of technology as intangible force; mystification of technological processes
  - got: cumulus-cloud, server-racks, tower-of-babel, jacquard-loom · wanted: cumulus-cloud
- **"General Intelligence" reinforces systems of centralised control and power**
  - concepts (llm): centralised control over knowledge; power concentration and suppression; systemic dominance and hierarchy; limitation of individual autonomy
  - got: panopticon, open-pit-mine, submarine-cable, switchboard · wanted: *panopticon*, *tower-of-babel*
- **AI is not a singular intelligent entity; it's an ecosystem of people, machines, and places**
  - concepts (llm): interconnected systems without central control; collaboration across diverse entities; dynamic interaction between human and machine; distributed intelligence without singular origin
  - got: cumulus-cloud, mycelium, earth-at-night, american-progress · wanted: *mycelium*, *honeycomb*, *murmuration*
- **The frontier narrative was – and remains – a colonial strategy to justify exploiting, erasing, pillaging, and **
  - concepts (llm): Justification of domination through narrative; Erasure of indigenous histories and identities; Violent transformation of landscapes and societies; Use of storytelling to legitimize exploitation
  - got: american-progress, open-pit-mine, tower-of-babel, fjord-coastline · wanted: american-progress, *open-pit-mine*, *old-world-map*
- **Enframing or "Gestell" refers to the way modern technology frames the world as nothing more than a standing-re**
  - concepts (llm): Reduction of reality to resources; Dominance of utilitarian thinking; Objectification of nature and humanity; Technological control over existence
  - got: tower-of-babel, open-pit-mine, panopticon, old-world-map · wanted: *open-pit-mine*
- **Anyone who believes exponetial growth and go on forever in a finite world is either a madman or an economist. **
  - concepts (llm): conflict between growth and limits; rationality vs. unrealistic belief; economic theory and ecological reality; finite resources vs. infinite growth assumptions
  - got: tower-of-babel, allotment-garden, american-progress, open-pit-mine · wanted: *tower-of-babel*, *open-pit-mine*, *earth-at-night*
- **Frontier landscape paintings framed territories as untouched and uninhabited. The cloud narrative erases the r**
  - concepts (llm): erasure of labor and politics in digital systems; ideological framing of nature as untouched; tension between representation and reality; processes of territorial control and erasure
  - got: american-progress, open-pit-mine, panopticon, server-racks · wanted: american-progress, cumulus-cloud, open-pit-mine
- **The cloud isn't a cloud at all. It's an underground network of people, resources, cables, and datacenters tire**
  - concepts (llm): collaboration without centralized control; interconnected systems supporting shared access; invisible infrastructure enabling digital magic; distributed effort sustaining global connectivity
  - got: submarine-cable, tower-of-babel, cumulus-cloud, mycelium · wanted: submarine-cable, server-racks, cumulus-cloud, *mycelium*, *switchboard*
- **The idea of the Earth as a vast, empty frontier, ready to be divided and claimed has a long history. It contin**
  - concepts (llm): Historical legacy of expansion and ownership; Innovation framed through conquest and division; Continuity of ideological narratives over time; Conceptualizing progress as territorial claiming
  - got: american-progress, open-pit-mine, earth-at-night, peasant-wedding · wanted: old-world-map, american-progress
- **Just as colonial landscape paintings present land as empty spaces, ripe for conquest, mainstream AI narratives**
  - concepts (llm): erasure of cultural and environmental contexts; abstraction of data collection processes; reduction of complex realities to empty spaces; ideological framing of land and knowledge
  - got: american-progress, open-pit-mine, tower-of-babel, the-large-pool-of-bursa · wanted: american-progress, *old-world-map*
- **universalism determines which problems are seen as worth solving, whose knowledge is considered valid, and who**
  - concepts (llm): Deciding what problems are important; Determining whose knowledge is accepted; Influencing who shapes technological development; Setting standards for valid knowledge and participation
  - got: tower-of-babel, american-progress, earth-at-night, river-delta · wanted: *tower-of-babel*, *old-world-map*
- **The myth of general artificial intelligence limits our imaginations of what AI can be.**
  - concepts (llm): Limitations on creative potential through belief systems; Restriction of possibilities by dominant narratives; Influence of myths on technological imagination; Cultural shaping of future technological forms
  - got: american-progress, jacquard-loom, cumulus-cloud, the-large-pool-of-bursa · wanted: *tower-of-babel*
- **There is a push for generality in machine learning, I'm totally against that: we have to work context by conte**
  - concepts (llm): opposition to universal solutions; importance of localized approaches; context-specific knowledge; resistance to generalized frameworks
  - got: honeycomb, card-catalogue, murmuration, old-world-map · wanted: *allotment-garden*, *peasant-wedding*
- **Placefulness allows us to move beyond the reductive map that typically depict AI infrastructure and instead en**
  - concepts (llm): reimagining systems beyond control and domination; co-creation of sustainable futures through collective vision; mapping as a process of ethical and ecological responsibility; transforming representation into possibility and justice
  - got: earth-at-night, server-racks, old-world-map, panopticon · wanted: old-world-map, *allotment-garden*, *river-delta*
- **Small AI – it might be janky, but it's yours.**
  - concepts (llm): individual ownership in a decentralized system; imperfect tools shaped by personal use; collaboration without centralized control; adaptation through personal interaction
  - got: fable-of-the-hares-and-the-frogs, tower-of-babel, murmuration, mycelium · wanted: *allotment-garden*
- **Antifragile is a property of systems that grow stronger and more capable when exposed to stressors, shocks, vo**
  - concepts (llm): systems becoming stronger through stress; resilience through exposure to uncertainty; growth from disorder and mistakes; adaptation in the face of volatility
  - got: spider-web, mycelium, submarine-cable, fjord-coastline · wanted: *mycelium*, *murmuration*
- **Moving with nature offers an alternative to the mainstream narratives of AI existing in the cloud.**
  - concepts (llm): Alternative ways of thinking about intelligence; Decentralized systems of knowledge creation; Human-nature integration in technological design; Reimagining the role of AI in society
  - got: cumulus-cloud, mycelium, murmuration, fable-of-the-hares-and-the-frogs · wanted: cumulus-cloud, *murmuration*, *river-delta*
- **Generous interfaces invite visitors to browse and wander through an archive's whole collection instead of typi**
  - concepts (llm): Encouraging exploration over direct searching; Facilitating organic discovery in information spaces; Promoting engagement through accessibility; Balancing openness with structured navigation
  - got: old-world-map, card-catalogue, the-large-pool-of-bursa, honeycomb · wanted: card-catalogue, *spider-web*, *river-delta*
- **Behind every automated system there is hidden human labour: rows of workers routing, labelling and moderating **
  - concepts (llm): human effort behind technology; invisible labor in automation; collaboration without visible control; hidden contributions to systems
  - got: switchboard, server-racks, honeycomb, tower-of-babel · wanted: switchboard, *honeycomb*, *jacquard-loom*
