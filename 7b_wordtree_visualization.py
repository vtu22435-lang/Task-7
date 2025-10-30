"""Task 7b: WordTree Visualization - Network Analysis"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import networkx as nx
from collections import Counter
from matplotlib.patches import FancyBboxPatch

df = pd.read_csv('telecom_customer_call_records_100.csv')
print(f"Dataset: {df.shape[0]} records from {df['Place'].nunique()} cities")

# Create hierarchical wordtree text
def make_pattern(row):
    cat = ["Short","Medium","Long","VeryLong"][min(row['Call_Duration_sec']//500, 3) if row['Call_Duration_sec']<2500 else 3]
    return f"{row['Place']} {cat} {row['Tower_ID']}"
wordtree_text = ' . '.join(df.apply(make_pattern, axis=1))

# Create figure with 4 subplots
fig = plt.figure(figsize=(18, 12))
fig.suptitle('WordTree Text Network Analysis & Visualization\nTelecom Customer Call Records', fontsize=22, fontweight='bold', y=0.98)

# 1. Hierarchical word cloud
ax1 = plt.subplot(2,2,1)
wc1 = WordCloud(width=800, height=600, background_color='white', colormap='Set2', max_words=100, relative_scaling=0.6, random_state=42).generate(wordtree_text)
ax1.imshow(wc1, interpolation='bilinear')
ax1.axis('off')
ax1.set_title('Word Cloud - Hierarchical Text Pattern', fontsize=14, fontweight='bold')

# 2. City-focused word cloud
ax2 = plt.subplot(2,2,2)
wc2 = WordCloud(width=800, height=600, background_color='#f0f0f0', colormap='plasma', max_words=50, random_state=42).generate(' '.join(df['Place'].tolist()*2))
ax2.imshow(wc2, interpolation='bilinear')
ax2.axis('off')
ax2.set_title('City Distribution Word Cloud', fontsize=14, fontweight='bold')

# 3. Network graph
ax3 = plt.subplot(2,2,3)
G = nx.Graph()
for city, towers in df.groupby('Place')['Tower_ID'].apply(lambda x: list(x.unique()[:3])).items():
    G.add_node(city, t='city')
    for tower in towers: G.add_node(tower, t='tower'); G.add_edge(city, tower)
pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
nx.draw_networkx_nodes(G, pos, node_color=['#ff6b6b' if G.nodes[n].get('t')=='city' else '#4ecdc4' for n in G.nodes()], node_size=[800 if G.nodes[n].get('t')=='city' else 300 for n in G.nodes()], alpha=0.8, ax=ax3)
nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.5, edge_color='gray', ax=ax3)
nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold', ax=ax3)
ax3.set_title('WordTree Network Graph\n(Cities <-> Towers)', fontsize=14, fontweight='bold')
ax3.axis('off')

# 4. Statistics panel
ax4 = plt.subplot(2,2,4)
ax4.axis('off')
city_counts = df['Place'].value_counts()
dur_stats = {f"{n} ({r})": len(df[(df['Call_Duration_sec']>=l)&(df['Call_Duration_sec']<h)]) if h else len(df[df['Call_Duration_sec']>=l]) for n,r,l,h in [("Short","<500s",0,500),("Medium","500-1500s",500,1500),("Long","1500-2500s",1500,2500),("Very Long",">=2500s",2500,None)]}
stats = f"TEXT NETWORK STATISTICS\n{'='*40}\n\n[*] TOP 5 CITIES:\n" + '\n'.join(f"  {i}. {c:12s} {'|'*int(n/2)} {n}" for i,(c,n) in enumerate(city_counts.head(5).items(),1))
stats += f"\n\n[*] CALL DURATION:\n" + '\n'.join(f"  {c:20s} {'#'*int(n/len(df)*20)} {n:2d} ({n/len(df)*100:.1f}%)" for c,n in dur_stats.items())
stats += f"\n\n[*] KEY METRICS:\n  > Total: {len(df)}\n  > Cities: {df['Place'].nunique()}\n  > Towers: {df['Tower_ID'].nunique()}\n  > Avg Duration: {df['Call_Duration_sec'].mean():.0f}s\n  > Med Duration: {df['Call_Duration_sec'].median():.0f}s"
ax4.add_patch(FancyBboxPatch((0.05,0.05), 0.9, 0.9, boxstyle="round,pad=0.02", edgecolor='#2c3e50', facecolor='#ecf0f1', linewidth=2, transform=ax4.transAxes))
ax4.text(0.5, 0.5, stats, ha='left', va='center', transform=ax4.transAxes, fontsize=10, fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=1))
ax4.set_title('Statistical Analysis', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('7b_wordtree_visualization_output.png', dpi=300, bbox_inches='tight')
plt.close()

# Console output
word_freq = Counter(wordtree_text.split())
print("\nTOP 15 TERMS:", *[f"{w}: {c}" for w,c in word_freq.most_common(15)], sep="\n  ")
print("\nCITY ANALYSIS:", *[f"{city}: {len(df[df['Place']==city])} calls, Avg={df[df['Place']==city]['Call_Duration_sec'].mean():.0f}s" for city in df['Place'].unique()], sep="\n  ")
print("[SUCCESS] Task 7b completed!")
