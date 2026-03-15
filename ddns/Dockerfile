FROM alpine:3.20

RUN apk add --no-cache curl jq

COPY update.sh /usr/local/bin/ddns-update
RUN chmod +x /usr/local/bin/ddns-update

CMD ["ddns-update"]
