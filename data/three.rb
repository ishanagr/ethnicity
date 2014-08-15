#!/usr/bin/ruby

vowels = "AEIOU".split('')
eth = ARGV[0] || 'asian'
len = ARGV[1].to_i || 3
patt = ARGV[2] || ''
f = File.open("/Users/ishanagr/workspace/src/ethnicity/#{eth}.csv", "r")
h = {}
f.each_line do |line|
  line.strip!
  l = line.length
  if patt == 'cv'
    new_l = ""
    line.split("").each do |c|
      new_l += if vowels.include?(c) then 'v' else 'c' end
    end
    line = new_l
  elsif patt == 'lastl'
    line = line[-1]
    l = 3
  end
  for i in 0..(l-3)
    w = line[i..(i+len-1)]
    count = h[w]
    count = 0 if count==nil
    h[w] = count+1
  end
end
f.close
r = h.sort {|a1,a2| a2[1]<=>a1[1]}
f_name = "#{eth}_#{len}_#{patt}.csv" 
File.open(f_name, 'w') do |file| 
  for i in r
    puts "#{i[0]},#{i[1]}"
    file.write("#{i[0]},#{i[1]}\n")
  end
end
