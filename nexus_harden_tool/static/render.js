// render.js -- the Nexus web UI's markdown/code-block renderer, lifted verbatim (same look, same
// copy-button behaviour) so reports here read exactly like the chat UI. See tools/server.cpp's
// renderMD()/cpCode() if this ever needs to be re-synced with changes made there.
function partialMD(text){
  var fences=(text.match(/^```/gm)||[]).length;
  return fences%2!==0 ? text+'\n```' : text;
}
function renderMD(s){
  s=s.replace(/<0x[0-9A-Fa-f]{2}>/g,'').replace(/<\/s>|<s>|<unk>/g,'');
  s=s.replace(/�/g,'').replace(/​/g,'').replace(/‌/g,'').replace(/‍/g,'');
  s=s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  var codeBlocks=[];
  s=s.replace(/```([\w+-]*)\n?([\s\S]*?)```/g,(_,lang,code)=>{
    var id='cb'+Math.random().toString(36).slice(2);
    var html='<div class="code-wrap"><div class="code-hdr"><span class="code-lang">'+(lang||'text')+'</span>'
      +'<button class="code-copy" id="'+id+'" onclick="cpCode(\''+id+'\',this)">Copy</button></div>'
      +'<pre><code>'+code.replace(/\n$/,'')+'</code></pre></div>';
    var idx=codeBlocks.length;
    codeBlocks.push(html);
    return '\x00CODEBLOCK'+idx+'\x00';
  });
  s=s.replace(/^\|(.+)\|\n\|[-| :]+\|\n((?:\|.+\|\n?)+)/gm,(_,hdr,rows)=>{
    var th=hdr.split('|').map(c=>'<th>'+c.trim()+'</th>').join('');
    var tr=rows.trim().split('\n').map(r=>'<tr>'+r.replace(/^\||\|$/g,'').split('|').map(c=>'<td>'+c.trim()+'</td>').join('')+'</tr>').join('');
    return '<table><thead><tr>'+th+'</tr></thead><tbody>'+tr+'</tbody></table>';
  });
  s=s.replace(/^&gt;\s?(.+)$/gm,'<blockquote>$1</blockquote>');
  s=s.replace(/^#{3}\s+(.+)$/gm,'<h3>$1</h3>');
  s=s.replace(/^#{2}\s+(.+)$/gm,'<h2>$1</h2>');
  s=s.replace(/^#\s+(.+)$/gm,'<h1>$1</h1>');
  s=s.replace(/^---+$/gm,'<hr>');
  s=s.replace(/`([^`\n]+)`/g,'<code>$1</code>');
  s=s.replace(/\*\*\*(.+?)\*\*\*/g,'<strong><em>$1</em></strong>');
  s=s.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
  s=s.replace(/\*([^\n*]+?)\*/g,'<em>$1</em>');
  s=s.replace(/~~(.+?)~~/g,'<del>$1</del>');
  s=s.replace(/\[([^\]]+)\]\(([^)]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
  s=s.replace(/^[-*+]\s+(.+)$/gm,'<li>$1</li>');
  s=s.replace(/(<li>[\s\S]*?<\/li>)(\n<li>[\s\S]*?<\/li>)*/g, function(m){ return '<ul>'+m+'</ul>'; });
  var blocks=s.split(/\n\n+/);
  var out = blocks.map(b=>{
    b=b.trim();if(!b)return'';
    if(/^\x00CODEBLOCK\d+\x00$/.test(b))return b;
    if(/^<(h[1-6]|ul|ol|pre|hr|table|blockquote|div)/.test(b))return b;
    return '<p>'+b.replace(/\n/g,'<br>')+'</p>';
  }).filter(Boolean).join('\n');
  out=out.replace(/\x00CODEBLOCK(\d+)\x00/g,(_,i)=>codeBlocks[+i]);
  return out;
}
function cpCode(id,btn){
  var code=document.getElementById(id).closest('.code-wrap').querySelector('code').textContent;
  navigator.clipboard.writeText(code).then(()=>{
    btn.textContent='Copied!';btn.classList.add('ok');
    setTimeout(()=>{btn.textContent='Copy';btn.classList.remove('ok');},1400);
  });
}
